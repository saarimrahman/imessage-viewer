import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from twin.export import (
    BUBBLE,
    CONTEXT_TURNS,
    ME,
    OPENER_PROMPT,
    SYSTEM,
    coerce_chat,
    collapse_turns,
    examples_from_turns,
    fit_messages,
    is_alternating,
    is_assistant_row,
    opener_for,
    parse_person_arg,
    partition_examples,
    person_id_for,
    resolve_subject,
    sessionize,
    system_for,
    write_dataset,
)
from twin.eval import chr_f, ranking_score, rouge_l, score_reply
from twin.retrieve import retrieve
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
from twin.train import (
    DEFAULT_MODEL,
    MODELS,
    adapter_dir,
    checkpoint_id,
    hash_train_file,
    has_adapter,
    list_adapter_runs,
    make_run_id,
    note_reference_eval,
    parse_checkpoint_id,
    restore_best_checkpoint,
    save_every_for,
    steps_for_examples,
    train_command,
    write_run_metadata,
)
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
            [{"role": t["role"], "content": t["content"]} for t in turns],
            [
                {"role": "user", "content": f"you free{BUBBLE}tonight?"},
                {"role": "assistant", "content": f"yeah{BUBBLE}wait 20"},
            ],
        )

    def test_skips_empty_bubbles(self):
        turns = collapse_turns([(False, "  "), (True, "ok")])

        self.assertEqual(
            [{"role": t["role"], "content": t["content"]} for t in turns],
            [{"role": "assistant", "content": "ok"}],
        )

    def test_same_sender_after_a_long_gap_starts_a_new_turn(self):
        gap = 31 * 60 * 1_000_000_000
        turns = collapse_turns([(True, "yesterday", 0), (True, "today", gap)])

        self.assertEqual(
            [{"role": t["role"], "content": t["content"]} for t in turns],
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

    def test_keeps_a_thread_you_started_out_of_reply_training(self):
        rows = examples_from_turns(
            [{"role": "assistant", "content": "omw"}],
            system="sys",
        )

        self.assertEqual(rows, [])

    def test_drops_a_leading_assistant_turn_from_the_window(self):
        turns = [
            {"role": "assistant", "content": "left on read"},
            {"role": "user", "content": "sorry"},
            {"role": "assistant", "content": "all good"},
        ]

        rows = examples_from_turns(turns, system="sys")

        roles = [m["role"] for m in rows[0]["messages"]]
        self.assertEqual(roles, ["system", "user", "assistant"])
        self.assertEqual(rows[0]["messages"][1]["content"], "sorry")

    def test_keeps_one_character_sent_text(self):
        rows = examples_from_turns(
            [{"role": "user", "content": "coming?"}, {"role": "assistant", "content": "k"}],
            system="sys",
        )

        self.assertEqual(rows[0]["messages"][-1]["content"], "k")

    def test_writes_one_example_per_target(self):
        turns = [
            {"role": "user", "content": "what are you doing"},
            {"role": "assistant", "content": "walking home"},
            {"role": "user", "content": "want food"},
            {"role": "assistant", "content": "yeah"},
        ]

        rows = examples_from_turns(turns, system="sys", augment=True)

        self.assertEqual(len(rows), 2)
        self.assertEqual([m["content"] for m in rows[-1]["messages"][-2:]], ["want food", "yeah"])
        self.assertTrue(all(is_alternating(row["messages"]) for row in rows))


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

    def test_opener_is_not_mixed_into_reply_examples(self):
        rows = examples_from_turns(
            [{"role": "assistant", "content": "omw"}],
            system="sys",
            opener="text as Alex",
        )
        self.assertEqual(rows, [])


class DatasetWriterTest(unittest.TestCase):
    def test_holdout_rows_are_not_copied_into_training(self):
        examples = [
            {
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": str(i)},
                    {"role": "assistant", "content": str(i)},
                ],
                "_sid": i,
                "_date": i,
                "_query": str(i),
                "_reply": str(i),
            }
            for i in range(20)
        ]
        with tempfile.TemporaryDirectory() as out:
            n_train, n_valid, n_test = write_dataset(examples, out)
            with open(out + "/train.jsonl", encoding="utf-8") as f:
                train = [json.loads(line) for line in f]
            with open(out + "/valid.jsonl", encoding="utf-8") as f:
                valid = [json.loads(line) for line in f]
            with open(out + "/test.jsonl", encoding="utf-8") as f:
                test = [json.loads(line) for line in f]

        self.assertEqual(n_train + n_valid + n_test, 20)
        self.assertGreater(n_valid, 0)
        self.assertGreater(n_test, 0)
        train_replies = {row["messages"][-1]["content"] for row in train}
        valid_replies = {row["messages"][-1]["content"] for row in valid}
        test_replies = {row["messages"][-1]["content"] for row in test}
        self.assertFalse(train_replies & valid_replies)
        self.assertFalse(train_replies & test_replies)
        self.assertFalse(valid_replies & test_replies)


class SessionAndFitTest(unittest.TestCase):
    def test_session_gap_stops_context_from_crossing_days(self):
        day = 24 * 60 * 60 * 1_000_000_000
        turns = [
            {"role": "user", "content": "old", "_date": 0},
            {"role": "assistant", "content": "ok", "_date": 1},
            {"role": "user", "content": "new", "_date": day},
            {"role": "assistant", "content": "hey", "_date": day + 1},
        ]
        sessions = sessionize(turns)
        self.assertEqual(len(sessions), 2)
        first = examples_from_turns(sessions[1], system="sys")
        self.assertEqual(first[0]["messages"][1]["content"], "new")
        self.assertNotIn("old", [m["content"] for m in first[0]["messages"]])

    def test_fit_messages_drops_oldest_context_and_keeps_the_target(self):
        class FakeTok:
            def apply_chat_template(self, messages, return_dict=False, **kwargs):
                return list(range(sum(len(m["content"]) for m in messages)))

        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "aaaaaaaaaa"},
            {"role": "assistant", "content": "bbbbbbbbbb"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "yo"},
        ]
        fitted = fit_messages(messages, FakeTok(), max_seq_length=20)
        self.assertEqual([m["content"] for m in fitted], ["sys", "hi", "yo"])

    def test_rejects_a_target_that_cannot_fit(self):
        class FakeTok:
            def apply_chat_template(self, messages, return_dict=False, **kwargs):
                return list(range(1000))

        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "yo"},
        ]
        self.assertIsNone(fit_messages(messages, FakeTok(), max_seq_length=10))

    def test_same_session_targets_stay_in_one_split(self):
        examples = [
            {"messages": [], "_sid": "a", "_date": 1, "_reply": "one"},
            {"messages": [], "_sid": "a", "_date": 2, "_reply": "two"},
            {"messages": [], "_sid": "b", "_date": 10, "_reply": "three"},
            {"messages": [], "_sid": "c", "_date": 20, "_reply": "four"},
        ]
        split = partition_examples(examples)
        grouped = {key: {row["_sid"] for row in rows} for key, rows in split.items()}
        self.assertIn("a", grouped["train"] | grouped["valid"] | grouped["test"])
        self.assertEqual(len([row for rows in split.values() for row in rows if row["_sid"] == "a"]), 2)
        homes = {name for name, sids in grouped.items() if "a" in sids}
        self.assertEqual(len(homes), 1)

    def test_inference_context_budget_matches_training(self):
        from twin.job import HISTORY_TURNS

        self.assertEqual(HISTORY_TURNS, CONTEXT_TURNS)
        self.assertEqual(CONTEXT_TURNS, 10)

    def test_coerce_chat_merges_consecutive_same_role_turns(self):
        messages = coerce_chat(
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
                {"role": "user", "content": "there"},
                {"role": "assistant", "content": "yo"},
            ]
        )
        self.assertEqual(messages[1]["content"], f"hi{BUBBLE}there")
        self.assertTrue(is_alternating(messages))

    def test_retrieve_skips_the_current_query(self):
        index = [
            {"query": "you free", "reply": "yeah", "tokens": ["you", "free"]},
            {"query": "you around later", "reply": "maybe", "tokens": ["you", "around", "later"]},
        ]
        hits = retrieve("you free tonight", index, k=2, exclude=["you free"])
        self.assertEqual(hits[0]["query"], "you around later")

    def test_holdout_metrics_prefer_close_replies(self):
        good = score_reply("yeah give me 10", "yeah give me 10")
        bland = score_reply("Of course! I would be happy to help you with that.", "yeah give me 10")
        self.assertGreater(ranking_score(good), ranking_score(bland))
        self.assertGreater(chr_f("yeah 10", "yeah 10"), 0.9)
        self.assertGreater(rouge_l("yeah wait 20", "yeah wait 20"), 0.9)


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
            "adapter_runs", "run_id", "data_hash", "early_stopped",
        ):
            self.assertIn(key, s)
        self.assertEqual(s["person"], "me")
        self.assertIsInstance(s["busy"], bool)
        self.assertIsInstance(s["has_adapter"], bool)
        self.assertIsInstance(s["mlx"], bool)
        self.assertIsInstance(s["runs"], list)
        self.assertIsInstance(s["adapter_runs"], list)
        self.assertIn("early_stopped", s)
        self.assertEqual(len(s["step_seconds"]), 3)

    def test_brief_snapshot_skips_heavy_fields(self):
        s = snapshot(brief=True)
        self.assertNotIn("metrics", s)
        self.assertNotIn("models", s)
        self.assertNotIn("runs", s)
        self.assertNotIn("adapter_runs", s)
        self.assertIn("busy", s)
        self.assertIn("eta_seconds", s)

    def test_rejects_an_unknown_run_without_starting(self):
        ok, err = start_train(run="forever")
        self.assertFalse(ok)
        self.assertIn("run", err)

    def test_rejects_invalid_step_counts(self):
        ok, err = start_train(iters=0)
        self.assertFalse(ok)
        self.assertIn("steps", err)
        ok, err = start_train(iters="nope")
        self.assertFalse(ok)
        self.assertIn("steps", err)

    def test_rejects_a_malformed_resume_checkpoint(self):
        ok, err = start_train(resume_from="not-a-checkpoint")
        self.assertFalse(ok)
        self.assertIn("Checkpoint", err)

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
        self.assertEqual(len(MODELS), 20)
        self.assertTrue(MODELS["qwen3-capable"]["recommended"])
        self.assertFalse(MODELS["qwen35-4b"].get("recommended"))
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
        for key in ("g9v3-3b", "minicpm5-1b", "ministral3-3b", "qwen3-8b", "qwen35-4b"):
            self.assertIn(key, MODELS)
        self.assertEqual(MODELS["qwen3-8b"]["layers"], 4)
        self.assertEqual(MODELS["llama31-8b"]["batch_size"], 1)

    def test_complete_steps_cover_every_example(self):
        self.assertEqual(steps_for_examples(101, 4), 26)
        self.assertEqual(steps_for_examples(101, 4, epochs=3), 78)

    def test_training_command_reports_loss_and_holdout_loss(self):
        cmd = train_command("model", "data", "adapter", 100, steps_per_report=5, steps_per_eval=20)
        self.assertIn("--mask-prompt", cmd)
        self.assertIn("mlx_lora.py", cmd[2])
        self.assertEqual(cmd[cmd.index("--steps-per-report") + 1], "5")
        self.assertEqual(cmd[cmd.index("--steps-per-eval") + 1], "20")
        self.assertEqual(cmd[cmd.index("--save-every") + 1], "50")
        self.assertEqual(cmd[cmd.index("--grad-accumulation-steps") + 1], "1")
        self.assertEqual(cmd[cmd.index("--learning-rate") + 1], "1e-05")

    def test_training_command_can_resume_from_weights(self):
        cmd = train_command(
            "model", "data", "adapter", 200,
            save_every=25,
            resume_adapter_file="/tmp/adapters.safetensors",
        )
        self.assertEqual(cmd[cmd.index("--save-every") + 1], "25")
        self.assertEqual(cmd[cmd.index("--resume-adapter-file") + 1], "/tmp/adapters.safetensors")

    def test_long_runs_save_periodic_checkpoints(self):
        self.assertEqual(save_every_for(30), 30)
        self.assertEqual(save_every_for(20000), 1000)

    def test_reference_plateau_stops_after_patience(self):
        tracker = {}
        self.assertFalse(note_reference_eval(tracker, 2.0, 1, patience=3, min_evals=3))
        self.assertFalse(note_reference_eval(tracker, 1.5, 10, patience=3, min_evals=3))
        self.assertFalse(note_reference_eval(tracker, 1.495, 20, patience=3, min_evals=3))
        self.assertFalse(note_reference_eval(tracker, 1.50, 30, patience=3, min_evals=3))
        self.assertTrue(note_reference_eval(tracker, 1.51, 40, patience=3, min_evals=3))
        self.assertEqual(tracker["best_iter"], 10)
        self.assertEqual(tracker["best"], 1.5)

    def test_tiny_reference_drops_do_not_reset_patience(self):
        tracker = {}
        note_reference_eval(tracker, 1.50, 10, min_delta=0.01, patience=2, min_evals=2)
        note_reference_eval(tracker, 1.495, 20, min_delta=0.01, patience=2, min_evals=2)
        self.assertTrue(note_reference_eval(tracker, 1.494, 30, min_delta=0.01, patience=2, min_evals=2))
        self.assertEqual(tracker["best_iter"], 10)

    def test_restore_best_checkpoint_copies_the_winning_save(self):
        with tempfile.TemporaryDirectory() as tmp:
            latest = os.path.join(tmp, "adapters.safetensors")
            best = os.path.join(tmp, "0000100_adapters.safetensors")
            later = os.path.join(tmp, "0000200_adapters.safetensors")
            with open(best, "w", encoding="utf-8") as f:
                f.write("best")
            with open(later, "w", encoding="utf-8") as f:
                f.write("later")
            with open(latest, "w", encoding="utf-8") as f:
                f.write("later")
            restore_best_checkpoint(tmp, 100)
            with open(latest, encoding="utf-8") as f:
                self.assertEqual(f.read(), "best")


class TwinAdapterRunTest(unittest.TestCase):
    def test_run_ids_include_time_hash_and_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_id = make_run_id(1_724_000_000, "abc123def456", 12400, tmp)
        self.assertIn("abc123de", run_id)
        self.assertTrue(run_id.endswith("-12400"))
        self.assertRegex(run_id, r"^\d{8}-\d{6}-abc123def456-12400$")

    def test_run_ids_do_not_reuse_an_existing_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = make_run_id(1_724_000_000, "abc123def456", 10, tmp)
            os.mkdir(os.path.join(tmp, first))
            second = make_run_id(1_724_000_000, "abc123def456", 10, tmp)
        self.assertNotEqual(first, second)
        self.assertTrue(second.endswith("-2"))

    def test_checkpoint_ids_round_trip(self):
        value = checkpoint_id("qwen3-capable", "20260816-160000-ab12cd34ef56-500", "latest")
        model, run_id, step = parse_checkpoint_id(value)
        self.assertEqual(model, "qwen3-capable")
        self.assertEqual(run_id, "20260816-160000-ab12cd34ef56-500")
        self.assertEqual(step, "latest")

    def test_new_runs_do_not_overwrite_an_existing_adapter(self):
        import twin.train as train

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(train, "TWIN_DIR", tmp):
                root = adapter_dir("qwen3-capable")
                os.makedirs(root, exist_ok=True)
                legacy = os.path.join(root, "adapters.safetensors")
                with open(legacy, "w", encoding="utf-8") as f:
                    f.write("old")
                newer = os.path.join(root, "20260816-160000-ab12cd34ef56-80")
                os.makedirs(newer)
                with open(os.path.join(newer, "adapters.safetensors"), "w", encoding="utf-8") as f:
                    f.write("new")
                with open(os.path.join(newer, "0000080_adapters.safetensors"), "w", encoding="utf-8") as f:
                    f.write("ckpt")
                write_run_metadata(
                    newer,
                    {
                        "model": "qwen3-capable",
                        "run_id": "20260816-160000-ab12cd34ef56-80",
                        "created_at": 1_724_000_100,
                        "data_hash": "ab12cd34ef56",
                        "iters": 80,
                        "examples": 40,
                        "run": "complete",
                        "status": "ready",
                    },
                )
                self.assertTrue(has_adapter("qwen3-capable"))
                runs = list_adapter_runs("me", "qwen3-capable")
                by_id = {row["run_id"]: row for row in runs}
                self.assertEqual(len(runs), 2)
                self.assertIn("legacy", by_id)
                self.assertIn("20260816-160000-ab12cd34ef56-80", by_id)
                self.assertEqual(by_id["20260816-160000-ab12cd34ef56-80"]["data_hash"], "ab12cd34ef56")
                self.assertEqual(by_id["20260816-160000-ab12cd34ef56-80"]["checkpoints"][0]["step"], "latest")
                with open(legacy, encoding="utf-8") as f:
                    self.assertEqual(f.read(), "old")

    def test_train_file_hash_changes_with_the_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "train.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write('{"a":1}\n')
            first = hash_train_file(tmp)
            with open(path, "a", encoding="utf-8") as f:
                f.write('{"a":2}\n')
            second = hash_train_file(tmp)
        self.assertEqual(len(first), 12)
        self.assertNotEqual(first, second)


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
        self.assertIn('id="twinNewChat"', html)
        self.assertIn('name="twinchat"', html)
        self.assertIn('id="twinIters"', html)
        self.assertIn('id="twinResume"', html)
        self.assertIn("Fresh weights", html)
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
            "adapter_runs": [
                {
                    "id": "qwen3-capable/20260816-160000-ab12cd34ef56-80",
                    "model": "qwen3-capable",
                    "run_id": "20260816-160000-ab12cd34ef56-80",
                    "name": "Qwen 3 Capable",
                    "params": "4B",
                    "created_at": 1_724_000_000,
                    "data_hash": "ab12cd34ef56",
                    "iters": 80,
                    "examples": 40,
                    "run": "complete",
                    "resume_from": "",
                    "checkpoints": [
                        {
                            "id": "qwen3-capable/20260816-160000-ab12cd34ef56-80/latest",
                            "step": "latest",
                            "step_n": 80,
                        }
                    ],
                }
            ],
        }
        with patch("twin.job.snapshot", return_value=fake):
            html = render_twin()
        start = html.index('id="twinChatSelect"')
        chat_html = html[start:html.index("</select>", start)]
        self.assertIn("qwen3-capable", chat_html)
        self.assertIn("Qwen 3 Capable", chat_html)
        self.assertIn("ab12cd34", chat_html)
        self.assertNotIn("qwen3-compact", chat_html)
        self.assertNotIn("Qwen 3 Compact UNIQUE", chat_html)
        self.assertNotIn('id="twinChatPicker" hidden', html)
