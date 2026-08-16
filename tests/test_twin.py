import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from twin.export import (
    ME,
    OPENER_PROMPT,
    SYSTEM,
    collapse_turns,
    examples_from_turns,
    is_assistant_row,
    opener_for,
    parse_person_arg,
    person_id_for,
    resolve_subject,
    system_for,
    write_dataset,
)
from twin.job import (
    BUSY_PHASES,
    DOWNLOAD_RE,
    activity_label,
    estimate_eta,
    format_duration,
    parse_iter,
    parse_metric,
    snapshot,
    start_train,
    stop_train,
    chat as twin_chat,
)
from twin.train import DEFAULT_MODEL, MODELS, adapter_dir, steps_for_examples, train_command
from render import NAV, render_twin


class CollapseTurnsTest(unittest.TestCase):
    def test_merges_consecutive_bubbles_from_the_same_person(self):
        turns = collapse_turns(
            [
                (False, "you free"),
                (False, "tonight?"),
                (True, "yeah"),
                (True, "wait 20"),
            ]
        )

        self.assertEqual(
            turns,
            [
                {"role": "user", "content": "you free\ntonight?"},
                {"role": "assistant", "content": "yeah\nwait 20"},
            ],
        )

    def test_skips_empty_bubbles(self):
        turns = collapse_turns([(False, "  "), (True, "ok")])

        self.assertEqual(turns, [{"role": "assistant", "content": "ok"}])

    def test_same_sender_after_a_long_gap_starts_a_new_turn(self):
        gap = 31 * 60 * 1_000_000_000
        turns = collapse_turns([(True, "yesterday", 0), (True, "today", gap)])

        self.assertEqual(
            turns,
            [
                {"role": "assistant", "content": "yesterday"},
                {"role": "assistant", "content": "today"},
            ],
        )


class ExamplesFromTurnsTest(unittest.TestCase):
    def test_one_example_per_reply_you_sent(self):
        turns = [
            {"role": "user", "content": "hey"},
            {"role": "assistant", "content": "yo"},
            {"role": "user", "content": "coming?"},
            {"role": "assistant", "content": "yeah 20 min"},
        ]

        rows = examples_from_turns(turns, system="sys")

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["messages"][0], {"role": "system", "content": "sys"})
        self.assertEqual(rows[0]["messages"][-1]["content"], "yo")
        self.assertEqual(rows[1]["messages"][-1]["content"], "yeah 20 min")

    def test_keeps_a_thread_you_started_with_a_neutral_prompt(self):
        rows = examples_from_turns(
            [{"role": "assistant", "content": "omw"}],
            system="sys",
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["messages"][1], {"role": "user", "content": OPENER_PROMPT})
        self.assertEqual(rows[0]["messages"][-1]["content"], "omw")

    def test_drops_a_leading_assistant_turn_from_the_window(self):
        turns = [
            {"role": "assistant", "content": "left on read"},
            {"role": "user", "content": "sorry"},
            {"role": "assistant", "content": "all good"},
        ]

        rows = examples_from_turns(turns, system="sys")

        roles = [m["role"] for m in rows[1]["messages"]]
        self.assertEqual(roles, ["system", "user", "assistant"])
        self.assertEqual(rows[1]["messages"][1]["content"], "sorry")

    def test_keeps_one_character_sent_text(self):
        rows = examples_from_turns(
            [{"role": "user", "content": "coming?"}, {"role": "assistant", "content": "k"}],
            system="sys",
        )

        self.assertEqual(rows[0]["messages"][-1]["content"], "k")

    def test_adds_a_distinct_short_context_variant(self):
        turns = [
            {"role": "user", "content": "what are you doing"},
            {"role": "assistant", "content": "walking home"},
            {"role": "user", "content": "want food"},
            {"role": "assistant", "content": "yeah"},
        ]

        rows = examples_from_turns(turns, system="sys", augment=True)

        self.assertEqual(len(rows), 3)
        self.assertEqual([m["content"] for m in rows[-1]["messages"][-2:]], ["want food", "yeah"])


class TwinPersonTest(unittest.TestCase):
    def test_you_is_the_default_subject(self):
        self.assertEqual(parse_person_arg(None), ME)
        self.assertEqual(parse_person_arg("me"), ME)
        self.assertEqual(resolve_subject(ME)["name"], "You")
        self.assertEqual(person_id_for("me"), ME)

    def test_phone_formats_collapse_to_the_same_person(self):
        self.assertEqual(person_id_for("5555550100"), person_id_for("+1 (555) 555-0100"))
        self.assertEqual(len(person_id_for("5555550100")), 16)
        self.assertNotEqual(person_id_for("5555550100"), ME)

    def test_hex_person_id_is_accepted(self):
        self.assertEqual(parse_person_arg("aabbccddeeff0011"), "aabbccddeeff0011")

    def test_unknown_handle_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_person_arg("not-a-contact")

    def test_assistant_rows_flip_when_training_a_contact(self):
        handle = "+15555550100"
        key = __import__("contacts").person_key(handle)
        self.assertTrue(is_assistant_row({"is_from_me": 1, "handle": None}))
        self.assertFalse(is_assistant_row({"is_from_me": 0, "handle": handle}))
        self.assertTrue(is_assistant_row({"is_from_me": 0, "handle": handle}, key))
        self.assertFalse(is_assistant_row({"is_from_me": 1, "handle": None}, key))

    def test_contact_prompt_stays_in_their_voice(self):
        self.assertEqual(system_for("You"), SYSTEM)
        self.assertIn("Alex", system_for("Alex"))
        self.assertNotIn("I would", system_for("Alex"))
        self.assertIn("Alex", opener_for("Alex"))
        self.assertEqual(opener_for("You"), OPENER_PROMPT)

    def test_contact_adapter_does_not_reuse_your_adapter_path(self):
        yours = adapter_dir("qwen3-capable")
        theirs = adapter_dir("qwen3-capable", "aabbccddeeff0011")
        self.assertTrue(yours.endswith("adapters/qwen3-capable"))
        self.assertIn("/people/aabbccddeeff0011/", theirs)
        self.assertNotEqual(yours, theirs)

    def test_opener_override_reaches_examples(self):
        rows = examples_from_turns(
            [{"role": "assistant", "content": "omw"}],
            system="sys",
            opener="text as Alex",
        )
        self.assertEqual(rows[0]["messages"][1], {"role": "user", "content": "text as Alex"})


class DatasetWriterTest(unittest.TestCase):
    def test_every_example_remains_in_training_when_reference_is_created(self):
        examples = [
            {"messages": [{"role": "user", "content": str(i)}, {"role": "assistant", "content": str(i)}]}
            for i in range(20)
        ]
        with tempfile.TemporaryDirectory() as out:
            n_train, n_reference = write_dataset(examples, out)
            with open(out + "/train.jsonl", encoding="utf-8") as f:
                written = [json.loads(line) for line in f]

        self.assertEqual(n_train, len(examples))
        self.assertGreater(n_reference, 0)
        self.assertCountEqual(written, examples)


class TwinJobTest(unittest.TestCase):
    def test_parse_iter_from_mlx_log_line(self):
        self.assertEqual(parse_iter("Iter 4: Train loss 5.597, Learning Rate 1.000e-04"), 4)
        self.assertIsNone(parse_iter("Loading datasets"))

    def test_parse_training_and_reference_metrics(self):
        train = parse_metric(
            "Iter 20: Train loss 1.231, Learning Rate 1e-4, Tokens/sec 245.5, Peak mem 4.321 GB"
        )
        valid = parse_metric("Iter 20: Val loss 1.456, Val took 2.0s")

        self.assertEqual(train["iter"], 20)
        self.assertEqual(train["train_loss"], 1.231)
        self.assertEqual(train["tokens_sec"], 245.5)
        self.assertEqual(train["memory_gb"], 4.321)
        self.assertEqual(valid, {"iter": 20, "reference_loss": 1.456})

    def test_snapshot_has_the_fields_the_ui_reads(self):
        s = snapshot()
        for key in (
            "phase", "detail", "iter", "iters", "examples", "busy", "has_adapter",
            "mlx", "metrics", "models", "eta_seconds", "elapsed_seconds",
            "phase_seconds", "step_seconds", "runs", "person", "person_name",
        ):
            self.assertIn(key, s)
        self.assertEqual(s["person"], "me")
        self.assertIsInstance(s["busy"], bool)
        self.assertIsInstance(s["has_adapter"], bool)
        self.assertIsInstance(s["mlx"], bool)
        self.assertIsInstance(s["runs"], list)
        self.assertEqual(len(s["step_seconds"]), 3)

    def test_brief_snapshot_skips_heavy_fields(self):
        s = snapshot(brief=True)
        self.assertNotIn("metrics", s)
        self.assertNotIn("models", s)
        self.assertNotIn("runs", s)
        self.assertIn("busy", s)
        self.assertIn("eta_seconds", s)

    def test_rejects_an_unknown_run_without_starting(self):
        ok, err = start_train(run="forever")
        self.assertFalse(ok)
        self.assertIn("run", err)

    def test_rejects_an_unknown_person_without_starting(self):
        ok, err = start_train(person_id="not-a-contact")
        self.assertFalse(ok)
        self.assertIn("contact", err.lower())

    def test_stop_does_nothing_when_idle(self):
        import twin.job as job

        with patch.object(job, "_runs", []):
            ok, err = stop_train()
        self.assertFalse(ok)
        self.assertIn("not training", err)

    def test_stop_clears_a_stale_running_row(self):
        import twin.job as job

        stale = {"status": "running", "name": "Qwen 2.5 Compact", "started_at": 1}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "runs.json")
            with patch.object(job, "RUNS_PATH", path), patch.object(job, "_runs", [stale]):
                ok, err = stop_train()
                self.assertTrue(ok)
                self.assertEqual(err, "")
                self.assertEqual(job._runs[0]["status"], "cancelled")

    def test_snapshot_marks_orphaned_runs_stopped(self):
        import twin.job as job

        stale = {"status": "running", "name": "Qwen 2.5 Compact", "started_at": 1}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "runs.json")
            with patch.object(job, "RUNS_PATH", path), patch.object(job, "_runs", [stale]):
                s = snapshot()
                self.assertEqual(s["runs"][0]["status"], "cancelled")
                self.assertFalse(s["busy"])

    def test_chat_passes_the_model_template_args(self):
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "PROMPT"
        mlx = MagicMock()
        mlx.generate.return_value = " yo "
        with patch("twin.job._load_model", return_value=(object(), tokenizer)):
            with patch.dict("sys.modules", {"mlx_lm": mlx}):
                reply = twin_chat("hey", model_key="qwen3-balanced")

        self.assertEqual(reply, "yo")
        self.assertEqual(
            tokenizer.apply_chat_template.call_args.kwargs["enable_thinking"],
            False,
        )

    def test_eta_uses_training_step_rate(self):
        self.assertIsNone(estimate_eta("loading", 10, 100, 1, 11))
        self.assertEqual(estimate_eta("training", 10, 100, 100, 110), 90)
        self.assertIsNone(estimate_eta("training", 0, 100, 100, 110))

    def test_duration_format_stays_compact(self):
        self.assertEqual(format_duration(9), "9s")
        self.assertEqual(format_duration(60), "1m")
        self.assertEqual(format_duration(75), "1m 15s")
        self.assertEqual(format_duration(3600), "1h")
        self.assertEqual(format_duration(3660), "1h 1m")

    def test_activity_label_names_the_current_step(self):
        self.assertEqual(activity_label({"busy": True, "phase": "downloading"}), "Twin · downloading")
        self.assertIn("left", activity_label({
            "busy": True,
            "phase": "training",
            "iter": 10,
            "iters": 100,
            "eta_seconds": 90,
        }))

    def test_download_log_lines_are_recognized(self):
        self.assertTrue(DOWNLOAD_RE.search("Fetching 12 files: 10%"))
        self.assertTrue(DOWNLOAD_RE.search("Downloading shards: 0%"))
        self.assertFalse(DOWNLOAD_RE.search("Loading pretrained model"))
        self.assertIn("downloading", BUSY_PHASES)
        self.assertIn("cancelling", BUSY_PHASES)


class TwinModelTest(unittest.TestCase):
    def test_offers_current_models_without_invalidating_existing_adapters(self):
        self.assertEqual(len(MODELS), 19)
        self.assertTrue(MODELS["qwen3-capable"]["recommended"])
        self.assertEqual(DEFAULT_MODEL, "qwen3-capable")
        self.assertEqual(MODELS["balanced"]["repo"], "mlx-community/Qwen2.5-1.5B-Instruct-4bit")

    def test_qwen3_chat_disables_reasoning_for_text_messages(self):
        self.assertEqual(
            MODELS["qwen3-balanced"]["chat_template_args"],
            {"enable_thinking": False},
        )

    def test_catalog_includes_non_qwen_mlx_models(self):
        self.assertEqual(MODELS["gemma3-compact"]["params"], "1B")
        self.assertEqual(
            MODELS["smollm3-capable"]["chat_template_args"],
            {"enable_thinking": False},
        )

    def test_catalog_includes_benchmark_and_8b_choices(self):
        for key in ("g9v3-3b", "minicpm5-1b", "ministral3-3b", "qwen3-8b"):
            self.assertIn(key, MODELS)
        self.assertEqual(MODELS["qwen3-8b"]["layers"], 4)
        self.assertEqual(MODELS["llama31-8b"]["batch_size"], 1)

    def test_complete_steps_cover_every_example(self):
        self.assertEqual(steps_for_examples(101, 4), 26)

    def test_training_command_reports_loss_and_reference_loss(self):
        cmd = train_command("model", "data", "adapter", 100, steps_per_report=5, steps_per_eval=20)
        self.assertIn("--mask-prompt", cmd)
        self.assertEqual(cmd[cmd.index("--steps-per-report") + 1], "5")
        self.assertEqual(cmd[cmd.index("--steps-per-eval") + 1], "20")


class TwinPageTest(unittest.TestCase):
    def test_nav_includes_twin(self):
        self.assertIn(("/twin", "Twin", "twin"), NAV)

    def test_page_has_the_hooks_app_js_needs(self):
        html = render_twin()
        self.assertIn('id="twinPage"', html)
        self.assertIn('id="twinTrain"', html)
        self.assertIn('id="twinCompose"', html)
        self.assertIn('id="twinThread"', html)
        self.assertIn('id="twinDataGrid"', html)
        self.assertIn('id="twinLossChart"', html)
        self.assertIn('name="twinmodel"', html)
        self.assertIn('id="twinModelSelect"', html)
        self.assertIn('<optgroup label="Maximum quality on 16 GB">', html)
        self.assertIn('id="twinTabs"', html)
        self.assertIn('id="twinPanelAudit"', html)
        self.assertIn('id="twinPanelModel"', html)
        self.assertIn('id="twinMetrics"', html)
        self.assertIn('id="twinProgress"', html)
        self.assertIn('id="twinProgressMeta"', html)
        self.assertIn('role="progressbar"', html)
        self.assertIn('id="twinSteps"', html)
        self.assertIn('id="twinStop"', html)
        self.assertIn('id="twinRuns"', html)
        self.assertIn('id="twinRunList"', html)
        self.assertIn('id="twinRunsTitle"', html)
        self.assertIn("Training attempts", html)
        self.assertNotIn("Recent runs", html)
        self.assertIn('class="twin-step-time"', html)
        self.assertIn('class="nav-dot"', html)
        self.assertIn('id="twinPanelChat"', html)
        self.assertIn('id="twinWho"', html)
        self.assertIn('id="twinWhoBtn"', html)
        self.assertIn('id="twinWhoList"', html)
        self.assertIn('data-person="me"', html)
        self.assertIn('id="twinChatSelect"', html)
        self.assertIn('id="twinChatPicker"', html)
        self.assertIn('name="twinchat"', html)
        self.assertIn('href="#audit"', html)
        self.assertIn('data-tab="audit"', html)
        self.assertNotIn('id="twinPanelSignals"', html)
        self.assertNotIn('id="twinTabSignals"', html)

    def test_chat_select_lists_only_trained_models(self):
        models = [
            {
                "key": "qwen3-capable",
                "name": "Qwen 3 Capable",
                "params": "4B",
                "download": "~2.3 GB",
                "memory": "ok",
                "description": "desc",
                "publisher": "Alibaba",
                "category": "Best overall",
                "recommended": True,
                "has_adapter": True,
                "cached": True,
            },
            {
                "key": "qwen3-compact",
                "name": "Qwen 3 Compact UNIQUE",
                "params": "0.6B",
                "download": "~350 MB",
                "memory": "ok",
                "description": "desc",
                "publisher": "Alibaba",
                "category": "Efficient alternatives",
                "recommended": False,
                "has_adapter": False,
                "cached": False,
            },
        ]
        fake = {
            "phase": "ready",
            "detail": "",
            "iter": 0,
            "iters": 0,
            "examples": 0,
            "busy": False,
            "has_adapter": True,
            "mlx": True,
            "metrics": [],
            "models": models,
            "model": "qwen3-capable",
            "eta_seconds": None,
            "elapsed_seconds": 12,
            "phase_seconds": {},
            "step_seconds": [0, 0, 0],
            "runs": [],
        }
        with patch("twin.job.snapshot", return_value=fake):
            html = render_twin()
        start = html.index('id="twinChatSelect"')
        chat_html = html[start:html.index("</select>", start)]
        self.assertIn("qwen3-capable", chat_html)
        self.assertIn("Qwen 3 Capable", chat_html)
        self.assertNotIn("qwen3-compact", chat_html)
        self.assertNotIn("Qwen 3 Compact UNIQUE", chat_html)
        self.assertNotIn('id="twinChatPicker" hidden', html)
