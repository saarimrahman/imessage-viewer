(function () {
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const THEME_KEY = "theme";

  function formatDuration(seconds) {
    if (!Number.isFinite(seconds) || seconds < 0) return "";
    const total = Math.round(seconds);
    if (total < 60) return total + "s";
    const minutes = Math.floor(total / 60);
    const sec = total % 60;
    if (minutes < 60) return sec ? minutes + "m " + String(sec).padStart(2, "0") + "s" : minutes + "m";
    const hours = Math.floor(minutes / 60);
    const rest = minutes % 60;
    return rest ? hours + "h " + rest + "m" : hours + "h";
  }

  function activityLabel(s) {
    if (!s.busy) return "";
    const eta = formatDuration(s.eta_seconds);
    if (s.phase === "training") {
      const core = s.iters
        ? "Twin · " + (s.iter || 0).toLocaleString() + "/" + s.iters.toLocaleString()
        : "Twin · training";
      return eta ? core + " · " + eta + " left" : core;
    }
    return ({
      inspecting: "Twin · auditing",
      exporting: "Twin · building pairs",
      downloading: "Twin · downloading",
      loading: "Twin · loading",
      cancelling: "Twin · stopping",
    })[s.phase] || s.detail || "Twin · training";
  }

  function applyTwinActivity(s) {
    const nav = document.querySelector('.nav-link[href="/twin"]');
    if (nav) nav.classList.toggle("is-training", !!s.busy);
    const right = document.querySelector(".topbar-right");
    let chip = document.getElementById("twinChip");
    if (s.busy) {
      if (!chip && right) {
        chip = document.createElement("a");
        chip.id = "twinChip";
        chip.className = "twin-chip";
        chip.href = "/twin#model";
        right.prepend(chip);
      }
      if (chip) {
        chip.hidden = false;
        chip.textContent = activityLabel(s);
      }
    } else if (chip) {
      chip.remove();
    }
  }

  function systemTheme() {
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }

  function storedTheme() {
    const stored = localStorage.getItem(THEME_KEY);
    return stored === "light" || stored === "dark" ? stored : null;
  }

  function applyTheme(theme) {
    document.documentElement.dataset.theme = theme;
    document.documentElement.style.colorScheme = theme;
    const btn = document.getElementById("themeToggle");
    if (!btn) return;
    const dark = theme === "dark";
    btn.setAttribute("aria-pressed", dark ? "true" : "false");
    btn.setAttribute("aria-label", dark ? "Switch to light mode" : "Switch to dark mode");
  }

  function initTheme() {
    applyTheme(storedTheme() || systemTheme());
    const btn = document.getElementById("themeToggle");
    if (btn) {
      btn.addEventListener("click", () => {
        const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
        localStorage.setItem(THEME_KEY, next);
        applyTheme(next);
      });
    }
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
      if (!storedTheme()) applyTheme(systemTheme());
    });
  }

  function initChatList() {
    const filter = document.getElementById("filter");
    if (!filter) return;
    const known = document.getElementById("knownOnly");
    const sort = document.getElementById("sortSelect");

    function apply() {
      const q = filter.value.toLowerCase();
      const knownOnly = known && known.checked;
      document.querySelectorAll("#rows .chat-row").forEach((row) => {
        const matchesText = (row.dataset.search || "").includes(q);
        const matchesKnown = !knownOnly || row.dataset.known === "1";
        row.hidden = !(matchesText && matchesKnown);
      });
    }

    filter.addEventListener("input", apply);
    if (known) known.addEventListener("change", apply);

    function go() {
      const params = new URLSearchParams();
      if (sort) params.set("sort", sort.value);
      const start = document.querySelector('input[name="start"]:checked');
      if (start) params.set("start", start.value);
      location.href = "/?" + params.toString();
    }

    if (sort) sort.addEventListener("change", go);
    document.querySelectorAll('input[name="start"]').forEach((el) => {
      el.addEventListener("change", go);
    });
  }

  function initDatepicker() {
    const el = document.getElementById("datepicker");
    if (!el || !el.dataset.chatId) return;
    el.addEventListener("change", () => {
      if (el.value) location.href = "/chat/" + el.dataset.chatId + "?date=" + el.value;
    });
  }

  function initHeatmapFocus() {
    const wrap = document.querySelector(".heatmap-wrap");
    const dp = document.getElementById("datepicker");
    if (!wrap || !dp || !dp.value) return;
    const cell = wrap.querySelector('.hcell[data-day="' + dp.value + '"]');
    if (!cell) return;
    const r = cell.getBoundingClientRect();
    const w = wrap.getBoundingClientRect();
    wrap.scrollLeft += r.left + r.width / 2 - (w.left + w.width / 2);
  }

  function initChatHeatmap() {
    const slot = document.getElementById("heatmapSlot");
    if (!slot || !slot.dataset.chatId) return;
    fetch("/chat/" + slot.dataset.chatId + "/heatmap")
      .then((res) => (res.ok ? res.text() : ""))
      .then((html) => {
        slot.classList.remove("is-loading");
        if (!html) {
          slot.remove();
          return;
        }
        slot.innerHTML = html;
        initHeatMode();
        initHeatmapFocus();
      })
      .catch(() => {
        slot.classList.remove("is-loading");
        slot.remove();
      });
  }

  function initCountup() {
    document.querySelectorAll(".countup").forEach((el) => {
      const target = parseFloat(el.dataset.count);
      if (!isFinite(target)) return;
      const decimals = (el.dataset.count || "").includes(".") ? 1 : 0;
      if (reduceMotion) {
        el.textContent = target.toLocaleString(undefined, {
          minimumFractionDigits: decimals,
          maximumFractionDigits: decimals,
        });
        return;
      }
      const dur = 900;
      const start = performance.now();
      function step(now) {
        const p = Math.min(1, (now - start) / dur);
        const value = target * (1 - Math.pow(1 - p, 3));
        el.textContent = value.toLocaleString(undefined, {
          minimumFractionDigits: decimals,
          maximumFractionDigits: decimals,
        });
        if (p < 1) requestAnimationFrame(step);
      }
      requestAnimationFrame(step);
    });
    document.querySelectorAll(".lbbar").forEach((el) => {
      if (reduceMotion) {
        el.style.width = el.dataset.target;
        return;
      }
      requestAnimationFrame(() =>
        requestAnimationFrame(() => {
          el.style.width = el.dataset.target;
        })
      );
    });
  }

  function initTileDurations() {
    // One capture-phase listener, because "loadedmetadata" does not bubble and
    // a grid holds thousands of tiles.
    document.addEventListener(
      "loadedmetadata",
      (e) => {
        const video = e.target;
        if (!video.parentElement) return;
        const out = video.parentElement.querySelector(".tile-dur");
        if (!out || !isFinite(video.duration)) return;
        const total = Math.round(video.duration);
        const secs = total % 60;
        out.textContent = Math.floor(total / 60) + ":" + String(secs).padStart(2, "0");
      },
      true
    );
  }

  function initMediaSize() {
    const seg = document.getElementById("mediaSize");
    if (!seg) return;
    const root = document.documentElement;

    // The grid reflows on a size change. Hold the month that the reader looks
    // at, so the view does not jump to another year.
    function anchor() {
      for (const sec of document.querySelectorAll(".media-month")) {
        const top = sec.getBoundingClientRect().top;
        if (top > -sec.offsetHeight + 40) return { sec, top };
      }
      return null;
    }

    seg.querySelectorAll('input[name="mediasize"]').forEach((el) => {
      el.checked = el.value === (root.dataset.mediasize || "m");
      el.addEventListener("change", () => {
        const held = anchor();
        localStorage.setItem("mediasize", el.value);
        root.dataset.mediasize = el.value;
        if (held) window.scrollTo(0, held.sec.offsetTop - held.top);
        window.dispatchEvent(new Event("resize"));
      });
    });
  }

  function initMediaRail() {
    const sections = Array.from(document.querySelectorAll(".media-month"));
    const rail = document.getElementById("mediaRail");
    if (!sections.length || !rail) return;
    const track = document.getElementById("mediaRailTrack");
    const dot = document.getElementById("mediaRailDot");
    const label = document.getElementById("mediaRailLabel");
    let dragging = false;

    function docHeight() {
      return Math.max(1, document.documentElement.scrollHeight - window.innerHeight);
    }

    function layoutTicks() {
      track.querySelectorAll(".media-rail-tick").forEach((t) => t.remove());
      const total = document.documentElement.scrollHeight;
      let lastYear = null;
      sections.forEach((sec) => {
        const year = sec.dataset.year;
        if (year !== lastYear) {
          lastYear = year;
          const tick = document.createElement("div");
          tick.className = "media-rail-tick";
          tick.style.top = (sec.offsetTop / total) * 100 + "%";
          tick.textContent = year;
          track.appendChild(tick);
        }
      });
    }

    function sectionAtFrac(frac) {
      const targetTop = frac * docHeight();
      let cur = sections[0];
      for (const sec of sections) {
        if (sec.offsetTop <= targetTop + 60) cur = sec;
        else break;
      }
      return cur;
    }

    function updateDot() {
      const frac = window.scrollY / docHeight();
      dot.style.top = frac * 100 + "%";
    }

    let ticking = false;
    window.addEventListener("scroll", () => {
      if (!ticking) {
        requestAnimationFrame(() => {
          updateDot();
          ticking = false;
        });
        ticking = true;
      }
    });
    window.addEventListener("resize", layoutTicks);

    track.addEventListener("mousemove", (e) => {
      const rect = track.getBoundingClientRect();
      const frac = Math.min(1, Math.max(0, (e.clientY - rect.top) / rect.height));
      const sec = sectionAtFrac(frac);
      label.textContent = sec.dataset.label;
      label.style.top = frac * 100 + "%";
      rail.classList.add("active");
      if (dragging) window.scrollTo(0, frac * docHeight());
    });
    track.addEventListener("mouseleave", () => {
      if (!dragging) rail.classList.remove("active");
    });
    track.addEventListener("mousedown", (e) => {
      dragging = true;
      const rect = track.getBoundingClientRect();
      const frac = Math.min(1, Math.max(0, (e.clientY - rect.top) / rect.height));
      window.scrollTo(0, frac * docHeight());
    });
    window.addEventListener("mouseup", () => {
      dragging = false;
    });

    window.addEventListener("load", () => {
      layoutTicks();
      updateDot();
    });
    layoutTicks();
    updateDot();
  }

  function initChatRail() {
    const rail = document.getElementById("chatRail");
    if (!rail) return;
    const track = document.getElementById("chatRailTrack");
    const dot = document.getElementById("chatRailDot");
    const label = document.getElementById("chatRailLabel");
    const chatId = rail.dataset.chatId;
    const curDate = rail.dataset.current;
    const t0 = Date.parse(rail.dataset.min + "T00:00:00");
    const t1 = Date.parse(rail.dataset.max + "T00:00:00");
    const span = Math.max(1, t1 - t0);
    let dragging = false;
    let lastFrac = 0;

    function fracOf(dateStr) {
      return Math.min(1, Math.max(0, (Date.parse(dateStr + "T00:00:00") - t0) / span));
    }
    function dateAt(frac) {
      const d = new Date(t0 + frac * span);
      const m = String(d.getMonth() + 1).padStart(2, "0");
      const day = String(d.getDate()).padStart(2, "0");
      return d.getFullYear() + "-" + m + "-" + day;
    }
    const startYear = new Date(t0).getFullYear();
    const endYear = new Date(t1).getFullYear();
    for (let y = startYear; y <= endYear; y++) {
      const frac = fracOf(y + "-01-01");
      const tick = document.createElement("div");
      tick.className = "media-rail-tick";
      tick.style.top = frac * 100 + "%";
      tick.textContent = y;
      track.appendChild(tick);
    }

    function setDot(frac) {
      dot.style.top = frac * 100 + "%";
    }
    function showLabel(frac) {
      lastFrac = frac;
      label.textContent = dateAt(frac);
      label.style.top = frac * 100 + "%";
      rail.classList.add("active");
    }
    function fracFromEvent(e) {
      const rect = track.getBoundingClientRect();
      return Math.min(1, Math.max(0, (e.clientY - rect.top) / rect.height));
    }

    setDot(fracOf(curDate));

    track.addEventListener("mousemove", (e) => {
      showLabel(fracFromEvent(e));
    });
    track.addEventListener("mouseleave", () => {
      if (!dragging) rail.classList.remove("active");
    });
    track.addEventListener("mousedown", (e) => {
      dragging = true;
      showLabel(fracFromEvent(e));
      e.preventDefault();
    });
    window.addEventListener("mouseup", () => {
      if (!dragging) return;
      dragging = false;
      rail.classList.remove("active");
      const next = dateAt(lastFrac);
      if (next !== curDate) location.href = "/chat/" + chatId + "?date=" + next;
    });
  }

  function initLightbox() {
    const defaultChatId = document.body.dataset.chatId || null;
    const SELECTOR = ".tile, img.att, video.att";

    function mediaInfo(el) {
      const inner = el.classList.contains("tile") ? el.querySelector("img,video") : el;
      if (!inner) return null;
      return {
        src: inner.getAttribute("data-full-src") || el.getAttribute("data-full-src") || inner.getAttribute("src"),
        isVideo: inner.tagName === "VIDEO",
        msgId: el.dataset.msgId || inner.dataset.msgId,
        chatId: el.dataset.chatId || inner.dataset.chatId || defaultChatId,
        node: el,
      };
    }

    function collectItems() {
      return Array.from(document.querySelectorAll(SELECTOR)).map(mediaInfo).filter(Boolean);
    }

    const overlay = document.createElement("div");
    overlay.className = "lightbox-overlay";
    overlay.innerHTML =
      '<button type="button" class="lightbox-chat">View in chat</button>' +
      '<button class="lightbox-close" aria-label="Close">&times;</button>' +
      '<button class="lightbox-nav lightbox-prev" aria-label="Previous">&#8249;</button>' +
      '<div class="lightbox-content"></div>' +
      '<button class="lightbox-nav lightbox-next" aria-label="Next">&#8250;</button>';
    document.body.appendChild(overlay);
    const content = overlay.querySelector(".lightbox-content");
    const chatBtn = overlay.querySelector(".lightbox-chat");

    const menu = document.createElement("div");
    menu.className = "ctx-menu";
    menu.innerHTML = '<div class="ctx-menu-item" id="ctxShowInChat">Show in chat</div>';
    document.body.appendChild(menu);
    let menuMsgId = null;
    let menuChatId = null;
    let items = [];
    let curIndex = 0;

    function chatUrl(chat, msg) {
      if (!chat || !msg) return null;
      return "/chat/" + chat + "?around=" + msg;
    }

    function render(i) {
      curIndex = (i + items.length) % items.length;
      const it = items[curIndex];
      content.innerHTML = it.isVideo
        ? '<video src="' + it.src + '" controls autoplay></video>'
        : '<img src="' + it.src + '">';
      chatBtn.hidden = !chatUrl(it.chatId, it.msgId);
    }

    function openAt(node) {
      items = collectItems();
      const idx = items.findIndex((it) => it.node === node);
      if (idx === -1) return;
      render(idx);
      overlay.classList.add("open");
    }

    function close() {
      overlay.classList.remove("open");
      content.innerHTML = "";
    }

    overlay.querySelector(".lightbox-close").addEventListener("click", close);
    overlay.addEventListener("click", (e) => {
      if (e.target === overlay) close();
    });
    overlay.querySelector(".lightbox-prev").addEventListener("click", () => render(curIndex - 1));
    overlay.querySelector(".lightbox-next").addEventListener("click", () => render(curIndex + 1));
    chatBtn.addEventListener("click", () => {
      const it = items[curIndex];
      const url = it && chatUrl(it.chatId, it.msgId);
      if (url) location.href = url;
    });
    document.addEventListener("keydown", (e) => {
      if (!overlay.classList.contains("open")) return;
      if (e.key === "Escape") close();
      else if (e.key === "ArrowLeft") render(curIndex - 1);
      else if (e.key === "ArrowRight") render(curIndex + 1);
    });

    document.body.addEventListener("click", (e) => {
      const el = e.target.closest(SELECTOR);
      if (!el || !mediaInfo(el)) return;
      e.preventDefault();
      openAt(el);
    });

    document.body.addEventListener("contextmenu", (e) => {
      const el = e.target.closest(SELECTOR);
      if (!el || !mediaInfo(el)) return;
      e.preventDefault();
      const info = mediaInfo(el);
      menuMsgId = info.msgId;
      menuChatId = info.chatId;
      menu.style.left = e.clientX + "px";
      menu.style.top = e.clientY + "px";
      menu.classList.add("open");
    });

    document.getElementById("ctxShowInChat").addEventListener("click", () => {
      const url = chatUrl(menuChatId, menuMsgId);
      if (url) location.href = url;
    });
    document.addEventListener("click", (e) => {
      if (!menu.contains(e.target)) menu.classList.remove("open");
    });
  }

  function initTapbacks() {
    const pop = document.createElement("div");
    pop.className = "tapback-pop";
    document.body.appendChild(pop);
    let openFor = null;

    function close() {
      pop.classList.remove("open");
      openFor = null;
    }

    function open(btn) {
      let detail;
      try {
        detail = JSON.parse(btn.dataset.detail || "[]");
      } catch (err) {
        return;
      }
      pop.innerHTML = detail
        .map(() => '<div class="tapback-pop-row"><span class="emoji"></span><span class="who"></span><span class="verb"></span></div>')
        .join("");
      Array.from(pop.children).forEach((row, i) => {
        row.querySelector(".emoji").textContent = detail[i].emoji;
        row.querySelector(".who").textContent = detail[i].who;
        row.querySelector(".verb").textContent = detail[i].verb;
      });
      pop.classList.add("open");
      const r = btn.getBoundingClientRect();
      const w = pop.offsetWidth;
      const h = pop.offsetHeight;
      let left = r.left + r.width / 2 - w / 2;
      left = Math.max(8, Math.min(left, window.innerWidth - w - 8));
      let top = r.bottom + 8;
      if (top + h > window.innerHeight - 8) top = Math.max(8, r.top - h - 8);
      pop.style.left = left + "px";
      pop.style.top = top + "px";
      openFor = btn;
    }

    document.body.addEventListener("click", (e) => {
      const btn = e.target.closest(".tapbacks");
      if (btn) {
        e.preventDefault();
        if (openFor === btn) close();
        else open(btn);
        return;
      }
      if (!pop.contains(e.target)) close();
    });

    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") close();
    });
    window.addEventListener("scroll", close, { passive: true });
    window.addEventListener("resize", close);
  }

  // Each cell carries all three counts. The buckets are relative to the mode
  // in view, so a quiet sent-only day still separates from an empty one.
  function initHeatMode() {
    const seg = document.getElementById("heatMode");
    const card = document.querySelector(".heatmap-card");
    if (!seg || !card) return;
    const maxes = {};
    ["a", "r", "s"].forEach((k, i) => {
      maxes[k] = Number(card.dataset.max.split(",")[i]) || 1;
    });
    const cells = card.querySelectorAll(".hcell[data-day]");
    const nouns = { a: "message", r: "received", s: "sent" };

    function apply(mode) {
      const top = maxes[mode];
      cells.forEach((cell) => {
        const n = Number(cell.dataset[mode]) || 0;
        let level = 0;
        if (n > 0) level = n <= top * 0.25 ? 1 : n <= top * 0.5 ? 2 : n <= top * 0.75 ? 3 : 4;
        cell.className = cell.className.replace(/heat-\d/, "heat-" + level);
        const noun = nouns[mode];
        cell.title = cell.dataset.day + ": " + n + " " + noun + (mode === "a" && n !== 1 ? "s" : "");
      });
    }

    seg.querySelectorAll('input[name="heatmode"]').forEach((el) => {
      el.addEventListener("change", () => apply(el.value));
    });
  }

  function initTwin() {
    const page = document.getElementById("twinPage");
    if (!page) return;
    const statusEl = document.getElementById("twinStatus");
    const hintEl = page.querySelector(".twin-hint");
    const progressTrack = document.getElementById("twinProgress");
    const progressEl = document.querySelector("#twinProgress span");
    const progressMetaEl = document.getElementById("twinProgressMeta");
    const metricsEl = document.getElementById("twinMetrics");
    const trainBtn = document.getElementById("twinTrain");
    const stopBtn = document.getElementById("twinStop");
    const goChatBtn = document.getElementById("twinGoChat");
    const readoutEl = document.getElementById("twinMetricReadout");
    const itersInput = document.getElementById("twinIters");
    const resumeSelect = document.getElementById("twinResume");
    const sendBtn = document.getElementById("twinSend");
    const input = document.getElementById("twinInput");
    const thread = document.getElementById("twinThread");
    const form = document.getElementById("twinCompose");
    const chatPicker = document.getElementById("twinChatPicker");
    const chatSelect = document.getElementById("twinChatSelect");
    const newChatBtn = document.getElementById("twinNewChat");
    const TABS = ["audit", "model", "chat"];
    const tabButtons = Array.from(page.querySelectorAll(".twin-tab"));
    const panels = Array.from(page.querySelectorAll(".twin-panel"));
    const modelSelect = document.getElementById("twinModelSelect");
    const modelNameEl = document.getElementById("twinModelName");
    const modelMetaEl = document.getElementById("twinModelMeta");
    const modelCopyEl = document.getElementById("twinModelCopy");
    const modelMemoryEl = document.getElementById("twinModelMemory");
    const modelRecommendedEl = document.getElementById("twinModelRecommended");
    const modelDownloadedEl = document.getElementById("twinModelDownloaded");
    const modelTrainedEl = document.getElementById("twinModelTrained");
    const stepEls = Array.from(page.querySelectorAll("#twinSteps li"));
    const whoBtn = document.getElementById("twinWhoBtn");
    const whoMenu = document.getElementById("twinWhoMenu");
    const whoSearch = document.getElementById("twinWhoSearch");
    const whoList = document.getElementById("twinWhoList");
    const whoNameEl = document.getElementById("twinWhoName");
    const whoAvatarEl = document.getElementById("twinWhoAvatar");
    const PERSON_KEY = "twin-person";
    const WAITING_PHASES = ["inspecting", "exporting", "downloading", "loading"];
    const history = [];
    let pollTimer = null;
    let sending = false;
    let lastStatus = null;
    let wasBusy = page.classList.contains("is-busy");
    let people = [];
    let personId = page.dataset.person || "me";

    function selectedModel() {
      return modelSelect.value || "qwen3-capable";
    }

    function selectedModelInfo(s) {
      return modelsForPerson(s).find((model) => model.key === selectedModel());
    }

    function selectedPerson() {
      return people.find((person) => person.id === personId) || people[0] || { id: "me", name: "You", trained: [] };
    }

    function isYou(person) {
      return !person || person.id === "me";
    }

    function personTrained(modelKey) {
      const person = selectedPerson();
      return !!(person.trained && person.trained.indexOf(modelKey) >= 0);
    }

    function modelsForPerson(s) {
      const models = s.models || [];
      if (!people.length) return models;
      return models.map((model) => Object.assign({}, model, {
        has_adapter: personTrained(model.key),
      }));
    }

    function applyPersonCopy(person) {
      const you = isYou(person);
      const name = (person && person.name) || "You";
      const lede = document.getElementById("twinLede");
      const dataTitle = document.getElementById("twinDataTitle");
      const dataCopy = document.getElementById("twinDataCopy");
      const chatTitle = document.getElementById("twinChatTitle");
      if (lede) {
        lede.textContent = you
          ? "Fine-tune a private local model on the way you actually text. Messages and adapters never leave this Mac."
          : "Fine-tune a private local model on how " + name + " actually texts. Messages and adapters never leave this Mac.";
      }
      if (dataTitle) dataTitle.textContent = you ? "Your training material" : "Training material for " + name;
      if (dataCopy) {
        dataCopy.textContent = you
          ? "Direct 1:1 chats are used for training. Each reply is one example. Later sessions are held out so validation is real. Group chats are counted here and left out of the adapter. Media without text cannot teach a language model and is counted separately."
          : "Direct 1:1 chats with " + name + " are used for training. Each of their replies is one example. Later sessions are held out so validation is real. Group chats are counted here and left out of the adapter. Media without text cannot teach a language model and is counted separately.";
      }
      if (chatTitle) chatTitle.textContent = you ? "Text your twin" : "Text " + name;
      input.placeholder = you ? "Text the twin…" : "Text " + name + "…";
      if (whoNameEl) whoNameEl.textContent = name;
      if (whoAvatarEl && person && person.avatar) whoAvatarEl.innerHTML = person.avatar;
      page.dataset.person = person ? person.id : "me";
      emptyState();
    }

    function closeWho() {
      if (!whoMenu || whoMenu.hidden) return;
      whoMenu.hidden = true;
      if (whoBtn) whoBtn.setAttribute("aria-expanded", "false");
      page.classList.remove("is-who-open");
    }

    function renderWhoList() {
      if (!whoList) return;
      const q = (whoSearch && whoSearch.value ? whoSearch.value : "").trim().toLowerCase();
      whoList.replaceChildren();
      people.forEach((person) => {
        const hay = ((person.name || "") + " " + (person.handle || "")).toLowerCase();
        if (q && hay.indexOf(q) < 0) return;
        const li = document.createElement("li");
        li.className = "twin-who-option" + (person.id === personId ? " is-selected" : "");
        li.setAttribute("role", "option");
        li.setAttribute("aria-selected", person.id === personId ? "true" : "false");
        li.dataset.id = person.id;
        const avatar = document.createElement("span");
        avatar.className = "twin-who-avatar";
        avatar.innerHTML = person.avatar || "";
        const body = document.createElement("span");
        body.className = "twin-who-option-copy";
        const name = document.createElement("strong");
        name.textContent = person.name;
        const meta = document.createElement("span");
        const bits = [];
        if (person.texts) {
          bits.push(person.texts.toLocaleString() + (person.id === "me" ? " sent" : " texts"));
        }
        if (person.trained && person.trained.length) bits.push("trained");
        meta.textContent = bits.join(" · ");
        body.append(name, meta);
        li.append(avatar, body);
        whoList.appendChild(li);
      });
      if (!whoList.childElementCount) {
        const empty = document.createElement("li");
        empty.className = "twin-who-empty";
        empty.textContent = "No matching contacts.";
        whoList.appendChild(empty);
      }
    }

    function openWho() {
      if (!whoMenu || !whoBtn || whoBtn.disabled) return;
      whoMenu.hidden = false;
      whoBtn.setAttribute("aria-expanded", "true");
      page.classList.add("is-who-open");
      if (whoSearch) whoSearch.value = "";
      renderWhoList();
      if (whoSearch) whoSearch.focus();
    }

    function selectPerson(id, fromUser) {
      const person = people.find((item) => item.id === id);
      if (!person) return;
      const changed = person.id !== personId;
      personId = person.id;
      try { localStorage.setItem(PERSON_KEY, personId); } catch (err) {}
      applyPersonCopy(person);
      closeWho();
      if (fromUser && changed) {
        resetChat();
        loadDataProfile();
        if (lastStatus) applyStatus(lastStatus);
      }
    }

    function adaptersForPerson(s) {
      const person = selectedPerson();
      if (s.adapter_runs && s.person === (person.id || "me")) return s.adapter_runs || [];
      return (person && person.adapters) || [];
    }

    function runLabel(run) {
      const bits = [run.name || run.model || "Model"];
      if (run.params) bits.push(run.params);
      const when = formatWhen(run.created_at);
      if (when) bits.push(when);
      if (run.iters) bits.push(run.iters.toLocaleString() + " steps");
      if (run.data_hash) bits.push(run.data_hash.slice(0, 8));
      return bits.join(" · ");
    }

    function checkpointLabel(ckpt) {
      if (ckpt.step === "latest") {
        return ckpt.step_n ? "Latest · " + Number(ckpt.step_n).toLocaleString() + " steps" : "Latest";
      }
      return "Step " + Number(ckpt.step_n || ckpt.step).toLocaleString();
    }

    function fillCheckpointSelect(select, runs, selectedId, includeFresh) {
      const prev = selectedId == null ? select.value : selectedId;
      select.replaceChildren();
      if (includeFresh) {
        const fresh = document.createElement("option");
        fresh.value = "";
        fresh.textContent = "Fresh weights";
        select.appendChild(fresh);
      }
      runs.forEach((run) => {
        const ckpts = run.checkpoints || [];
        if (!ckpts.length) return;
        if (ckpts.length === 1) {
          const option = document.createElement("option");
          option.value = ckpts[0].id;
          option.textContent = runLabel(run);
          select.appendChild(option);
          return;
        }
        const group = document.createElement("optgroup");
        group.label = runLabel(run);
        ckpts.forEach((ckpt) => {
          const option = document.createElement("option");
          option.value = ckpt.id;
          option.textContent = checkpointLabel(ckpt);
          group.appendChild(option);
        });
        select.appendChild(group);
      });
      const values = Array.from(select.options).map((opt) => opt.value);
      if (prev && values.indexOf(prev) >= 0) select.value = prev;
      else if (includeFresh) select.value = "";
      else if (values.length) select.value = values[0];
    }

    function chatModel() {
      return chatSelect.value || "";
    }

    function chatModelInfo(s) {
      return modelsForPerson(s).find((model) => model.key === chatModel());
    }

    function showTab(name, fromUser) {
      if (name === "signals") name = "model";
      if (!TABS.includes(name)) name = page.dataset.tab || "audit";
      if (fromUser) page.classList.add("has-switched");
      page.dataset.tab = name;
      tabButtons.forEach((btn) => {
        btn.setAttribute("aria-selected", btn.dataset.tab === name ? "true" : "false");
      });
      panels.forEach((panel) => {
        panel.hidden = panel.dataset.tab !== name;
      });
      if (name === "chat") {
        requestAnimationFrame(() => { thread.scrollTop = thread.scrollHeight; });
      }
    }

    function tabFromLocation() {
      const hash = location.hash.replace("#", "");
      if (hash === "signals") return "model";
      if (TABS.includes(hash)) return hash;
      return page.dataset.tab || "audit";
    }

    function setHash(name) {
      if (location.hash.replace("#", "") === name) return;
      if (location.hash) location.hash = name;
      else window.history.replaceState(null, "", "#" + name);
    }

    function esc(text) {
      const d = document.createElement("div");
      d.textContent = text;
      return d.innerHTML;
    }

    function syncNewChat() {
      if (!newChatBtn) return;
      newChatBtn.disabled = sending || !thread.querySelector(".row");
    }

    function resetChat() {
      history.length = 0;
      thread.replaceChildren();
      emptyState();
    }

    function emptyState() {
      if (thread.querySelector(".row")) {
        syncNewChat();
        return;
      }
      const person = selectedPerson();
      const you = isYou(person);
      const trained = !!(chatPicker && !chatPicker.hidden && chatSelect.options.length);
      const key = String(trained) + ":" + (person.id || "me");
      const existing = thread.querySelector(".twin-empty");
      if (existing && existing.dataset.key === key) {
        syncNewChat();
        return;
      }
      if (existing) existing.remove();
      const p = document.createElement("p");
      p.className = "twin-empty";
      p.dataset.key = key;
      if (trained) {
        p.append(you
          ? "You are on the right; your twin replies on the left."
          : "You are on the right; " + person.name + " replies on the left.");
      } else {
        p.append(you
          ? "Train a model, then text it here. You are on the right; your twin replies on the left. "
          : "Train a model as " + person.name + ", then text them here. You are on the right; they reply on the left. ");
        const a = document.createElement("a");
        a.href = "#model";
        a.textContent = "Choose a model";
        p.append(a);
      }
      thread.appendChild(p);
      syncNewChat();
    }

    function addBubbles(who, text, pendingRow) {
      const parts = String(text || "").split(/※|<\|bubble\|>/).map((part) => part.trim()).filter(Boolean);
      if (!parts.length) {
        if (pendingRow) {
          pendingRow.querySelector(".bubble").textContent = "";
          pendingRow.removeAttribute("data-pending");
        }
        return;
      }
      if (pendingRow) {
        pendingRow.querySelector(".bubble").textContent = parts[0];
        pendingRow.removeAttribute("data-pending");
        parts.slice(1).forEach((part) => addBubble(who, part));
        return;
      }
      parts.forEach((part) => addBubble(who, part));
    }

    function addBubble(who, text, pending) {
      const empty = thread.querySelector(".twin-empty");
      if (empty) empty.remove();
      const last = thread.querySelector(".row:last-child");
      const groupStart = !last || !last.classList.contains(who);
      if (last) {
        if (last.classList.contains(who)) last.classList.remove("tail");
        else last.classList.add("tail");
      }
      const row = document.createElement("div");
      row.className = "row " + who + (groupStart ? " group-start" : "") + " tail is-new";
      if (pending) row.dataset.pending = "1";
      row.innerHTML = '<div class="bubble tail">' + esc(text) + "</div>";
      thread.appendChild(row);
      row.scrollIntoView({ block: "end", behavior: reduceMotion ? "auto" : "smooth" });
      syncNewChat();
      return row;
    }

    function svgNode(name, attrs, text) {
      const node = document.createElementNS("http://www.w3.org/2000/svg", name);
      Object.keys(attrs || {}).forEach((key) => node.setAttribute(key, attrs[key]));
      if (text !== undefined) node.textContent = text;
      return node;
    }

    function chartXY(point, key, width, height, maxX, maxY) {
      return [
        34 + (point.iter / Math.max(1, maxX)) * (width - 46),
        10 + (1 - point[key] / Math.max(0.001, maxY)) * (height - 36),
      ];
    }

    function chartPath(points, key, width, height, maxX, maxY) {
      return points.filter((point) => Number.isFinite(point[key])).map((point, index) => {
        const [x, y] = chartXY(point, key, width, height, maxX, maxY);
        return (index ? "L" : "M") + x.toFixed(1) + " " + y.toFixed(1);
      }).join(" ");
    }

    function chartKind(key) {
      if (key === "reference_loss") return "is-reference";
      if (key === "tokens_sec") return "is-speed";
      return "is-train";
    }

    function chartValueLabel(key, value) {
      if (key === "train_loss") return "train " + formatLoss(value);
      if (key === "reference_loss") return "val " + formatLoss(value);
      if (key === "tokens_sec") {
        return value.toLocaleString(undefined, { maximumFractionDigits: 0 }) + " tok/s";
      }
      if (key === "memory_gb") return value.toFixed(1) + " GB";
      return String(value);
    }

    function hideChartHover(svg) {
      const wrap = svg.closest(".twin-chart-plot");
      const tip = wrap && wrap.querySelector(".twin-chart-tip");
      const hover = svg.querySelector(".chart-hover");
      if (tip) tip.style.opacity = "0";
      if (hover) hover.replaceChildren();
      svg._twinHoverX = null;
    }

    function showChartHover(svg, clientX) {
      const state = svg._twinChart;
      const wrap = svg.closest(".twin-chart-plot");
      const tip = wrap && wrap.querySelector(".twin-chart-tip");
      const hover = svg.querySelector(".chart-hover");
      if (!state || !state.hits.length || !wrap) {
        hideChartHover(svg);
        return;
      }
      const box = svg.getBoundingClientRect();
      const wbox = wrap.getBoundingClientRect();
      const x = (clientX - box.left) * (state.width / Math.max(1, box.width));
      if (x < 34 || x > state.width - 12) {
        hideChartHover(svg);
        return;
      }
      let best = state.hits[0];
      let bestDist = Math.abs(best.x - x);
      state.hits.forEach((hit) => {
        const dist = Math.abs(hit.x - x);
        if (dist < bestDist) {
          best = hit;
          bestDist = dist;
        }
      });
      svg._twinHoverX = clientX;
      if (hover) {
        hover.replaceChildren(...best.dots.map((dot) => svgNode("circle", {
          cx: dot.x.toFixed(1),
          cy: dot.y.toFixed(1),
          r: "4.5",
          class: "chart-hover-dot " + dot.kind,
        })));
      }
      if (tip) {
        tip.textContent = best.label;
        tip.style.left = (box.left - wbox.left + best.x * box.width / state.width) + "px";
        tip.style.top = (box.top - wbox.top + best.topY * box.height / state.height) + "px";
        tip.style.opacity = "1";
      }
    }

    function bindChartHover(svg) {
      if (svg.dataset.hoverBound) return;
      svg.dataset.hoverBound = "1";
      svg.addEventListener("pointermove", (e) => showChartHover(svg, e.clientX));
      svg.addEventListener("pointerleave", () => hideChartHover(svg));
    }

    function drawChart(svg, metrics, keys, maxY) {
      const width = 520;
      const height = 190;
      const hoverX = svg._twinHoverX;
      const maxX = Math.max(1, ...metrics.map((point) => point.iter || 0));
      const values = metrics.flatMap((point) => keys.map((key) => point[key])).filter(Number.isFinite);
      const yTop = maxY || Math.max(1, ...values) * 1.12;
      const grid = svg.querySelector(".chart-grid");
      const labels = svg.querySelector(".chart-labels");
      const dots = svg.querySelector(".chart-dots");
      const hover = svg.querySelector(".chart-hover");
      grid.replaceChildren();
      labels.replaceChildren();
      if (dots) dots.replaceChildren();
      if (hover && hoverX == null) hover.replaceChildren();
      for (let i = 0; i <= 3; i += 1) {
        const y = 10 + (i / 3) * (height - 36);
        grid.appendChild(svgNode("line", { x1: 34, y1: y, x2: width - 12, y2: y }));
        labels.appendChild(svgNode("text", { x: 30, y: y + 3, "text-anchor": "end" }, (yTop * (1 - i / 3)).toFixed(yTop < 10 ? 1 : 0)));
      }
      labels.appendChild(svgNode("text", { x: 34, y: height - 4 }, "0"));
      labels.appendChild(svgNode("text", { x: width - 12, y: height - 4, "text-anchor": "end" }, maxX.toLocaleString()));
      const hits = [];
      metrics.forEach((point) => {
        const marks = [];
        const parts = [];
        keys.forEach((key) => {
          if (!Number.isFinite(point[key])) return;
          const [x, y] = chartXY(point, key, width, height, maxX, yTop);
          marks.push({ x, y, kind: chartKind(key) });
          parts.push(chartValueLabel(key, point[key]));
        });
        if (!marks.length) return;
        hits.push({
          x: marks[0].x,
          topY: Math.min(...marks.map((mark) => mark.y)),
          dots: marks,
          label: "step " + Number(point.iter || 0).toLocaleString() + " · " + parts.join(" · "),
        });
      });
      keys.forEach((key, index) => {
        const path = svg.querySelectorAll(".chart-line")[index];
        if (path) path.setAttribute("d", chartPath(metrics, key, width, height, maxX, yTop));
        if (!dots || !path || !path.classList.contains("chart-reference")) return;
        metrics.filter((point) => Number.isFinite(point[key])).forEach((point) => {
          const [x, y] = chartXY(point, key, width, height, maxX, yTop);
          dots.appendChild(svgNode("circle", {
            cx: x.toFixed(1),
            cy: y.toFixed(1),
            r: "3",
            class: "chart-dot is-reference",
          }));
        });
      });
      svg._twinChart = { width, height, hits };
      bindChartHover(svg);
      if (hoverX != null) showChartHover(svg, hoverX);
    }

    function latestMetric(metrics, key) {
      let value = null;
      (metrics || []).forEach((point) => {
        if (Number.isFinite(point[key])) value = point[key];
      });
      return value;
    }

    function formatRate(value) {
      if (!Number.isFinite(value) || value <= 0) return "";
      return value.toExponential(2).replace(/e\+?/, "e");
    }

    function readoutStat(label, value) {
      if (!value) return null;
      const wrap = document.createElement("div");
      const dd = document.createElement("dd");
      const dt = document.createElement("dt");
      dd.textContent = value;
      dt.textContent = label;
      wrap.append(dd, dt);
      return wrap;
    }

    function drawReadout(metrics) {
      if (!readoutEl) return;
      const train = latestMetric(metrics, "train_loss");
      const val = latestMetric(metrics, "reference_loss");
      const lr = latestMetric(metrics, "learning_rate");
      const tok = latestMetric(metrics, "tokens_sec");
      const it = latestMetric(metrics, "it_sec");
      const mem = latestMetric(metrics, "memory_gb");
      const tokens = latestMetric(metrics, "trained_tokens");
      const items = [
        readoutStat("train", Number.isFinite(train) ? formatLoss(train) : ""),
        readoutStat("val", Number.isFinite(val) ? formatLoss(val) : ""),
        readoutStat("lr", formatRate(lr)),
        readoutStat("tok/s", Number.isFinite(tok) ? tok.toLocaleString(undefined, { maximumFractionDigits: 0 }) : ""),
        readoutStat("it/s", Number.isFinite(it) ? it.toLocaleString(undefined, { maximumFractionDigits: 2 }) : ""),
        readoutStat("peak GB", Number.isFinite(mem) ? mem.toFixed(1) : ""),
        readoutStat("tokens", Number.isFinite(tokens) ? tokens.toLocaleString(undefined, { maximumFractionDigits: 0 }) : ""),
      ].filter(Boolean);
      readoutEl.replaceChildren(...items);
      readoutEl.hidden = items.length === 0;
    }

    function drawMetrics(metrics) {
      drawChart(document.getElementById("twinLossChart"), metrics, ["train_loss", "reference_loss"]);
      drawChart(document.getElementById("twinSpeedChart"), metrics, ["tokens_sec"]);
      const speeds = metrics.map((point) => point.tokens_sec).filter(Number.isFinite);
      const memory = metrics.map((point) => point.memory_gb).filter(Number.isFinite);
      document.getElementById("twinPeakSpeed").textContent = speeds.length
        ? "Peak " + Math.max(...speeds).toLocaleString(undefined, { maximumFractionDigits: 0 }) + " tok/s"
        : "Waiting for training";
      document.getElementById("twinPeakMemory").textContent = memory.length
        ? "Peak " + Math.max(...memory).toFixed(1) + " GB"
        : "";
      drawReadout(metrics);
    }

    const STEP_COPY = [
      "Count usable text without exposing it.",
      "Sessionize 1:1 chats and hold out later sessions.",
      "Download weights if needed, then train on your JSONL.",
      "Open chat when you want to try the adapter.",
    ];
    const STEP_LIVE = {
      inspecting: "Counting usable sent text…",
      exporting: "Building conversation pairs…",
      downloading: "Downloading the model weights…",
      loading: "Loading the model from disk…",
      training: "Retraining the adapter on your texts…",
      cancelling: "Stopping training…",
    };

    function applySteps(s, hasAdapter) {
      const phase = s.phase;
      const active = phase === "inspecting" ? 0
        : phase === "exporting" ? 1
          : (phase === "downloading" || phase === "loading" || phase === "training" || phase === "cancelling") ? 2
            : (phase === "ready" || hasAdapter) ? 3 : 0;
      const times = s.step_seconds || [];
      stepEls.forEach((el, index) => {
        el.classList.toggle("is-active", index === active && phase !== "idle");
        el.classList.toggle("is-done", index < active || (index === 3 && hasAdapter));
        const small = el.querySelector("small");
        if (small) {
          let live = STEP_LIVE[phase];
          if (index === active && phase === "training" && !isYou(selectedPerson())) {
            live = "Retraining the adapter on " + selectedPerson().name + "’s texts…";
          }
          small.textContent = (index === active && live) ? live : STEP_COPY[index];
        }
        const timeEl = el.querySelector(".twin-step-time");
        if (!timeEl) return;
        if (index === 3) {
          timeEl.textContent = phase === "ready" && s.elapsed_seconds ? formatDuration(s.elapsed_seconds) : "";
          return;
        }
        const secs = times[index] || 0;
        timeEl.textContent = (index < active || index === active) && secs >= 0.5 ? formatDuration(secs) : "";
      });
    }

    function latestLosses(metrics) {
      let train = null;
      let reference = null;
      (metrics || []).forEach((point) => {
        if (Number.isFinite(point.train_loss)) train = point.train_loss;
        if (Number.isFinite(point.reference_loss)) reference = point.reference_loss;
      });
      return { train, reference };
    }

    function progressFor(s, modelReady) {
      if (s.phase === "ready" || (modelReady && !s.busy)) return 100;
      if (s.phase === "training" && s.iters) return 28 + 72 * (s.iter / s.iters);
      if (s.phase === "loading") return 24;
      if (s.phase === "downloading") return 18;
      if (s.phase === "exporting") return 10;
      if (s.phase === "inspecting") return 4;
      return 0;
    }

    function formatLoss(value) {
      return value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    }

    function revealSignals(show) {
      if (!metricsEl) return;
      const opening = show && metricsEl.hidden;
      metricsEl.hidden = !show;
      metricsEl.classList.toggle("is-entering", opening);
    }

    function syncModelPicker(s) {
      modelsForPerson(s).forEach((model) => {
        const option = Array.from(modelSelect.options).find((item) => item.value === model.key);
        if (!option) return;
        option.textContent = model.name + " — " + model.params + " · " + model.download + (model.has_adapter ? " · trained" : "");
      });
    }

    function syncChatPicker(s, preferId) {
      const runs = adaptersForPerson(s);
      fillCheckpointSelect(chatSelect, runs, preferId == null ? chatSelect.value : preferId, false);
      chatPicker.hidden = runs.length === 0;
      chatSelect.disabled = !!s.busy || runs.length === 0;
    }

    function syncResumePicker(s) {
      if (!resumeSelect) return;
      const model = selectedModel();
      const runs = adaptersForPerson(s).filter((run) => run.model === model);
      fillCheckpointSelect(resumeSelect, runs, resumeSelect.value, true);
      resumeSelect.disabled = !!s.busy;
    }

    function updateModelSummary(model) {
      if (!model) return;
      modelNameEl.textContent = model.name;
      modelMetaEl.textContent = model.publisher + " · " + model.params + " parameters · " + model.download + " download";
      modelCopyEl.textContent = model.description;
      modelMemoryEl.textContent = model.memory;
      modelRecommendedEl.hidden = !model.recommended;
      modelDownloadedEl.hidden = !model.cached;
      modelTrainedEl.hidden = !model.has_adapter;
    }

    function formatWhen(epoch) {
      if (!Number.isFinite(epoch) || epoch <= 0) return "";
      const delta = Date.now() / 1000 - epoch;
      if (delta < 45) return "just now";
      if (delta < 3600) return Math.max(1, Math.round(delta / 60)) + "m ago";
      if (delta < 20 * 3600) return Math.max(1, Math.round(delta / 3600)) + "h ago";
      const d = new Date(epoch * 1000);
      const today = new Date();
      const startOfToday = new Date(today.getFullYear(), today.getMonth(), today.getDate()).getTime();
      if (d.getTime() >= startOfToday - 86400000 && d.getTime() < startOfToday) return "Yesterday";
      const opts = d.getFullYear() === today.getFullYear()
        ? { month: "short", day: "numeric" }
        : { month: "short", day: "numeric", year: "numeric" };
      return d.toLocaleDateString(undefined, opts);
    }

    function liveRun(row, live) {
      if (!live || !live.busy || row.status !== "running") return row;
      const losses = latestLosses(live.metrics);
      return Object.assign({}, row, {
        elapsed_seconds: live.elapsed_seconds,
        examples: live.examples,
        sent_texts: live.sent_texts,
        chats: live.chats,
        augmented: live.augmented,
        iter: live.iter,
        iters: live.iters,
        train_loss: losses.train,
        reference_loss: losses.reference,
      });
    }

    function runStat(label, value) {
      if (!value) return null;
      const wrap = document.createElement("div");
      const dd = document.createElement("dd");
      const dt = document.createElement("dt");
      dd.textContent = value;
      dt.textContent = label;
      wrap.append(dd, dt);
      return wrap;
    }

    function showTrainingScreen() {
      revealSignals(true);
      showTab("model", true);
      setHash("model");
    }

    function openRunProgress() {
      if (lastStatus && lastStatus.model) modelSelect.value = lastStatus.model;
      showTrainingScreen();
      if (lastStatus) applyStatus(lastStatus);
      const trainCard = page.querySelector(".twin-train");
      requestAnimationFrame(() => {
        (trainCard || metricsEl).scrollIntoView({
          block: "start",
          behavior: reduceMotion ? "auto" : "smooth",
        });
      });
    }

    function renderRuns(runs, live) {
      const box = document.getElementById("twinRuns");
      const list = document.getElementById("twinRunList");
      if (!box || !list) return;
      const rows = (runs || []).filter((row) => (row.person || "me") === personId);
      box.hidden = rows.length === 0;
      list.replaceChildren();
      rows.forEach((raw) => {
        const row = liveRun(raw, live);
        const status = row.status === "ready" && row.early_stopped ? "Plateau"
          : row.status === "ready" ? "Done"
          : row.status === "cancelled" ? "Stopped"
            : row.status === "running" ? "Running"
              : "Failed";
        const kind = row.status === "ready" ? "ready"
          : row.status === "cancelled" ? "cancelled"
            : row.status === "running" ? "running"
              : "error";
        const li = document.createElement("li");
        li.className = "twin-run is-" + kind;
        if (kind === "running") li.classList.add("is-openable");

        const head = document.createElement("div");
        head.className = "twin-run-head";
        const title = document.createElement("strong");
        title.textContent = row.name || row.model || "Model";
        head.append(title);
        const when = formatWhen(row.started_at);
        if (when) {
          const time = document.createElement("time");
          time.dateTime = Number.isFinite(row.started_at)
            ? new Date(row.started_at * 1000).toISOString()
            : "";
          time.textContent = when;
          head.append(time);
        }

        const sub = document.createElement("div");
        sub.className = "twin-run-sub";
        const badge = document.createElement("span");
        badge.className = "twin-run-status";
        badge.textContent = status;
        sub.append(badge);
        const kindLabel = document.createElement("span");
        kindLabel.textContent = row.run === "quick" ? "Quick" : row.run === "complete" ? "Complete" : (row.run || "Train");
        sub.append(kindLabel);
        if (row.data_hash) {
          const hash = document.createElement("span");
          hash.textContent = row.data_hash.slice(0, 8);
          sub.append(hash);
        }
        if (row.params) {
          const size = document.createElement("span");
          size.textContent = row.params;
          sub.append(size);
        }
        if (kind === "running") {
          const actions = document.createElement("div");
          actions.className = "twin-run-actions";
          const open = document.createElement("a");
          open.className = "twin-run-open";
          open.href = "#model";
          open.textContent = "View progress";
          open.setAttribute("aria-label", "View progress for " + (row.name || "this run"));
          const stop = document.createElement("button");
          stop.type = "button";
          stop.className = "btn twin-run-stop";
          stop.textContent = "Stop";
          stop.setAttribute("aria-label", "Stop training " + (row.name || "this run"));
          if (live && live.phase === "cancelling") stop.disabled = true;
          actions.append(open, stop);
          sub.append(actions);
        }

        const stats = document.createElement("dl");
        stats.className = "twin-run-stats";
        const elapsed = formatDuration(row.elapsed_seconds);
        const items = [
          runStat("duration", elapsed),
          runStat("examples", Number.isFinite(row.examples) && row.examples > 0
            ? row.examples.toLocaleString() : ""),
          runStat("train", Number.isFinite(row.train_loss) ? formatLoss(row.train_loss) : ""),
          runStat("ref", Number.isFinite(row.reference_loss) ? formatLoss(row.reference_loss) : ""),
        ];
        if (row.status === "ready" && row.iters) {
          items.splice(2, 0, runStat("steps", row.iters.toLocaleString()));
        } else if (row.status !== "ready" && row.iters) {
          items.splice(2, 0, runStat("steps", (row.iter || 0).toLocaleString() + "/" + row.iters.toLocaleString()));
        }
        items.filter(Boolean).forEach((item) => stats.appendChild(item));

        li.append(head, sub);
        if (stats.childElementCount) li.append(stats);
        if (kind === "error" && row.detail) {
          const err = document.createElement("p");
          err.className = "twin-run-error";
          err.textContent = row.detail;
          li.append(err);
        }
        list.appendChild(li);
      });
    }

    function applyStatus(s) {
      lastStatus = s;
      const busy = !!s.busy;
      const model = selectedModelInfo(s);
      const modelReady = !!(model && model.has_adapter);
      trainBtn.disabled = busy || !s.mlx;
      stopBtn.hidden = !busy;
      stopBtn.disabled = s.phase === "cancelling";
      modelSelect.disabled = busy;
      if (itersInput) itersInput.disabled = busy;
      if (resumeSelect) resumeSelect.disabled = busy;
      if (whoBtn) {
        whoBtn.disabled = busy;
        if (busy) closeWho();
      }
      syncModelPicker(s);
      updateModelSummary(model);
      syncResumePicker(s);
      const preferChat = wasBusy && !busy && s.model && s.run_id
        && (s.phase === "ready" || s.phase === "cancelled")
        ? s.model + "/" + s.run_id + "/latest"
        : null;
      syncChatPicker(s, preferChat);
      const chatReady = !!(chatSelect.value && !chatPicker.hidden);
      sendBtn.disabled = busy || !chatReady || sending;
      input.disabled = busy || !chatReady;
      if (goChatBtn) goChatBtn.hidden = busy || !chatReady;
      if (!s.mlx) {
        statusEl.textContent = "mlx-lm is not installed.";
      } else if (WAITING_PHASES.includes(s.phase) || s.phase === "training" || s.phase === "cancelling") {
        statusEl.textContent = s.detail;
      } else if (s.phase === "error") {
        statusEl.textContent = s.detail || "Training failed.";
      } else if (s.phase === "cancelled") {
        statusEl.textContent = "Training stopped.";
      } else if (modelReady && s.early_stopped) {
        statusEl.textContent = model.name + " adapter ready. Validation loss plateaued, so training stopped early.";
      } else if (modelReady) {
        statusEl.textContent = model.name + " adapter ready. Review the curves, then go to chat.";
      } else {
        statusEl.textContent = model ? model.name + " has not been trained yet." : "Choose a model.";
      }
      const losses = latestLosses(s.metrics);
      const bits = [];
      if (losses.train != null) bits.push("train " + formatLoss(losses.train));
      if (losses.reference != null) bits.push("val " + formatLoss(losses.reference));
      const eta = formatDuration(s.eta_seconds);
      if (eta) bits.push("~" + eta + " left");
      else if (busy && s.elapsed_seconds) bits.push(formatDuration(s.elapsed_seconds));
      else if (busy) bits.push("Step " + (s.phase === "inspecting" ? 1 : s.phase === "exporting" ? 2 : 3) + " of 4");
      else if (s.phase === "cancelled") bits.push("Stopped");
      else if (s.early_stopped) bits.push("Plateau");
      else if (modelReady) bits.push(s.elapsed_seconds ? formatDuration(s.elapsed_seconds) : "Ready");
      progressMetaEl.textContent = bits.join(" · ");
      const waiting = WAITING_PHASES.includes(s.phase);
      const progress = progressFor(s, modelReady);
      progressTrack.classList.toggle("is-waiting", waiting);
      progressTrack.setAttribute("aria-busy", busy ? "true" : "false");
      progressTrack.setAttribute("aria-valuenow", String(Math.round(Math.max(0, Math.min(100, progress)))));
      progressEl.style.width = waiting ? "" : Math.max(0, Math.min(100, progress)) + "%";
      applySteps(s, modelReady);
      revealSignals(busy || (s.metrics || []).length > 0 || s.phase === "error" || s.phase === "cancelled");
      drawMetrics(s.metrics || []);
      renderRuns(s.runs, s);
      applyTwinActivity(s);
      page.classList.toggle("is-busy", busy);
      page.classList.toggle("is-error", s.phase === "error");
      if (wasBusy && !busy && s.phase === "ready") {
        const person = selectedPerson();
        if (person && s.model && person.trained && person.trained.indexOf(s.model) < 0) {
          person.trained.push(s.model);
        }
        resetChat();
        showTab("model", true);
        setHash("model");
        revealSignals(true);
        requestAnimationFrame(() => {
          (metricsEl || page.querySelector(".twin-train")).scrollIntoView({
            block: "start",
            behavior: reduceMotion ? "auto" : "smooth",
          });
        });
        loadPeople();
      } else if (wasBusy && !busy) {
        loadPeople();
      }
      wasBusy = busy;
      emptyState();
      if (hintEl) {
        if (!s.mlx) {
          hintEl.innerHTML =
            "Install MLX training support in this app's virtualenv, then restart the server:" +
            "<pre>./.venv/bin/python -m pip install -r twin/requirements.txt</pre>";
        } else if (s.phase === "error") {
          hintEl.textContent = "The previous run stopped. Quick is the shortest way to verify the setup before retrying Complete.";
        } else if (s.phase === "cancelled") {
          hintEl.textContent = "Training was stopped. Saved checkpoints stay available to chat with or continue from.";
        } else if (busy && s.examples) {
          const who = isYou(selectedPerson()) ? "sent texts" : "texts";
          hintEl.textContent = s.examples.toLocaleString() + " train examples cover " + s.sent_texts.toLocaleString() + " " + who + " across " + s.chats.toLocaleString() + " direct chats.";
        } else if (s.early_stopped) {
          hintEl.textContent = "Holdout loss stopped improving, so training ended. Chat starts on the last checkpoint, which is rarely the best one. Try the earlier steps in the chat picker.";
        } else if (modelReady) {
          hintEl.textContent = "Each train writes a new adapter. Review the curves here, then go to chat when you want to try it.";
        } else {
          hintEl.textContent = "Quick uses 30 steps on a recent slice to check that training runs. Complete uses 1:1 sessions, a real holdout, and three epochs. Leave steps blank for that default.";
        }
      }
      if (busy && !pollTimer) {
        pollTimer = setInterval(refreshStatus, 800);
      }
      if (!busy && pollTimer) {
        clearInterval(pollTimer);
        pollTimer = null;
      }
    }

    async function loadPeople() {
      try {
        const res = await fetch("/twin/people");
        const data = await res.json();
        people = data.people || [];
        let stored = null;
        try { stored = localStorage.getItem(PERSON_KEY); } catch (err) {}
        const busyPerson = lastStatus && lastStatus.busy ? lastStatus.person : null;
        const initial = busyPerson || (stored && people.some((person) => person.id === stored) ? stored : "me");
        personId = people.some((person) => person.id === initial) ? initial : "me";
        applyPersonCopy(selectedPerson());
        renderWhoList();
        if (lastStatus) applyStatus(lastStatus);
      } catch (err) {}
    }

    async function loadDataProfile() {
      const note = document.getElementById("twinDataNote");
      try {
        const res = await fetch("/twin/data?person=" + encodeURIComponent(personId));
        const profile = await res.json();
        if (!res.ok) throw new Error(profile.error || "Cannot inspect messages");
        page.querySelectorAll("[data-stat]").forEach((el) => {
          const value = profile[el.dataset.stat];
          el.textContent = Number.isFinite(value) ? value.toLocaleString() : "—";
        });
        note.textContent = "Complete scans the full archive at training time.";
      } catch (err) {
        note.textContent = "Could not inspect the message archive.";
      }
    }

    async function refreshStatus() {
      try {
        const res = await fetch("/twin/status");
        applyStatus(await res.json());
      } catch (err) {
        statusEl.textContent = "Cannot reach the trainer.";
      }
    }

    async function stopTrain() {
      const buttons = [stopBtn, ...page.querySelectorAll(".twin-run-stop")];
      buttons.forEach((btn) => { btn.disabled = true; });
      try {
        const res = await fetch("/twin/stop", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: "{}",
        });
        const data = await res.json();
        applyStatus(data);
        if (!res.ok && data.error) statusEl.textContent = data.error;
      } catch (err) {
        buttons.forEach((btn) => { btn.disabled = false; });
        statusEl.textContent = "Could not stop training.";
      }
    }

    stopBtn.addEventListener("click", stopTrain);

    if (goChatBtn) {
      goChatBtn.addEventListener("click", () => {
        showTab("chat", true);
        setHash("chat");
        if (input && !input.disabled) input.focus();
      });
    }

    trainBtn.addEventListener("click", async () => {
      const run = (page.querySelector('input[name="twinrun"]:checked') || {}).value || "complete";
      const model = selectedModel();
      const rawIters = itersInput ? itersInput.value.trim() : "";
      const iters = rawIters === "" ? null : Number(rawIters);
      if (rawIters !== "" && (!Number.isInteger(iters) || iters < 1)) {
        statusEl.textContent = "Steps must be a whole number of 1 or more.";
        return;
      }
      const resume = resumeSelect ? resumeSelect.value : "";
      trainBtn.disabled = true;
      showTrainingScreen();
      try {
        const res = await fetch("/twin/train", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ run, model, person: personId, iters, resume: resume || undefined }),
        });
        const data = await res.json();
        applyStatus(data);
        if (!res.ok && data.error) statusEl.textContent = data.error;
      } catch (err) {
        trainBtn.disabled = false;
        statusEl.textContent = "Train request failed.";
      }
    });

    modelSelect.addEventListener("change", () => {
      if (lastStatus) applyStatus(lastStatus);
    });

    chatSelect.addEventListener("change", () => {
      resetChat();
      if (lastStatus) applyStatus(lastStatus);
    });

    if (newChatBtn) {
      newChatBtn.addEventListener("click", () => {
        if (sending || !thread.querySelector(".row")) return;
        resetChat();
        if (input && !input.disabled) input.focus();
      });
    }

    if (whoBtn) {
      whoBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        if (whoBtn.disabled) return;
        if (whoMenu.hidden) openWho();
        else closeWho();
      });
    }
    if (whoSearch) whoSearch.addEventListener("input", renderWhoList);
    if (whoList) {
      whoList.addEventListener("click", (e) => {
        const option = e.target.closest("[data-id]");
        if (!option) return;
        selectPerson(option.dataset.id, true);
      });
    }
    document.addEventListener("click", (e) => {
      if (!whoMenu || whoMenu.hidden) return;
      if (e.target.closest("#twinWho")) return;
      closeWho();
    });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && whoMenu && !whoMenu.hidden) {
        closeWho();
        if (whoBtn) whoBtn.focus();
      }
    });

    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const text = input.value.trim();
      if (!text || sendBtn.disabled || sending) return;
      sending = true;
      input.value = "";
      addBubble("me", text);
      const pending = addBubble("them", "…", true);
      sendBtn.disabled = true;
      syncNewChat();
      try {
        const res = await fetch("/twin/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text, history, adapter: chatModel(), person: personId }),
        });
        const data = await res.json();
        const bubble = pending.querySelector(".bubble");
        if (!res.ok) {
          bubble.textContent = data.error || "The twin could not reply.";
          pending.removeAttribute("data-pending");
        } else {
          addBubbles("them", data.reply, pending);
          history.push({ role: "user", content: text });
          history.push({ role: "assistant", content: data.reply });
        }
      } catch (err) {
        pending.querySelector(".bubble").textContent = "The twin could not reply.";
        pending.removeAttribute("data-pending");
      }
      sending = false;
      syncNewChat();
      await refreshStatus();
    });

    emptyState();
    drawMetrics([]);
    const startTab = tabFromLocation();
    showTab(startTab);
    if (location.hash.replace("#", "") !== startTab) {
      window.history.replaceState(null, "", "#" + startTab);
    }
    tabButtons.forEach((btn) => {
      btn.addEventListener("click", (e) => {
        e.preventDefault();
        showTab(btn.dataset.tab, true);
        setHash(btn.dataset.tab);
      });
    });
    page.querySelector(".twin-tabs").addEventListener("keydown", (e) => {
      if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
      e.preventDefault();
      const i = TABS.indexOf(page.dataset.tab);
      const next = TABS[(i + (e.key === "ArrowRight" ? 1 : TABS.length - 1)) % TABS.length];
      showTab(next, true);
      setHash(next);
      const focusBtn = tabButtons.find((btn) => btn.dataset.tab === next);
      if (focusBtn) focusBtn.focus();
    });
    window.addEventListener("hashchange", () => {
      page.classList.add("has-switched");
      showTab(tabFromLocation());
    });
    document.getElementById("twinRunList").addEventListener("click", (e) => {
      if (e.target.closest(".twin-run-stop")) {
        e.preventDefault();
        e.stopPropagation();
        if (!e.target.closest(".twin-run-stop").disabled) stopTrain();
        return;
      }
      if (!e.target.closest(".twin-run.is-openable")) return;
      e.preventDefault();
      openRunProgress();
    });
    loadPeople();
    loadDataProfile();
    refreshStatus();
  }

  function initCircles() {
    const page = document.getElementById("circlesPage");
    const dataEl = document.getElementById("circlesData");
    const viewport = document.getElementById("circlesViewport");
    const world = document.getElementById("circlesWorld");
    const svg = document.getElementById("circlesEdges");
    const nodeLayer = document.getElementById("circlesNodes");
    const sheet = document.getElementById("circlesSheet");
    const find = document.getElementById("circlesFind");
    if (!page || !dataEl || !viewport || !world || !svg || !nodeLayer || !sheet) return;

    const data = JSON.parse(dataEl.textContent);
    if (!data.groups || !data.groups.length) return;

    const SVG_NS = "http://www.w3.org/2000/svg";
    const OFFSET = 2000;
    const cam = { x: 0, y: 0, k: 1 };
    const nodes = [];
    const byId = {};
    let selected = null;
    let alpha = 1;
    let running = false;
    let userMoved = false;
    let drag = null;

    function applyCam() {
      world.style.transform = "translate(" + cam.x + "px, " + cam.y + "px) scale(" + cam.k + ")";
    }

    function toWorld(clientX, clientY) {
      const rect = viewport.getBoundingClientRect();
      return {
        x: (clientX - rect.left - cam.x) / cam.k,
        y: (clientY - rect.top - cam.y) / cam.k,
      };
    }

    function fit(glide) {
      let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
      nodes.forEach((n) => {
        minX = Math.min(minX, n.x - n.r);
        minY = Math.min(minY, n.y - n.r);
        maxX = Math.max(maxX, n.x + n.r);
        maxY = Math.max(maxY, n.y + n.r);
      });
      const bw = Math.max(maxX - minX, 80);
      const bh = Math.max(maxY - minY, 80);
      const rect = viewport.getBoundingClientRect();
      const pad = 96;
      cam.k = Math.min(1.2, Math.max(0.32, Math.min((rect.width - pad) / bw, (rect.height - pad) / bh)));
      cam.x = rect.width / 2 - ((minX + maxX) / 2) * cam.k;
      cam.y = rect.height / 2 - ((minY + maxY) / 2) * cam.k;
      if (glide && !reduceMotion) {
        world.classList.add("is-gliding");
        world.addEventListener("transitionend", () => world.classList.remove("is-gliding"), { once: true });
      }
      applyCam();
    }

    function panTo(n) {
      const rect = viewport.getBoundingClientRect();
      cam.k = Math.max(cam.k, 1);
      cam.x = rect.width / 2 - n.x * cam.k;
      cam.y = rect.height / 2 - n.y * cam.k;
      if (!reduceMotion) {
        world.classList.add("is-gliding");
        world.addEventListener("transitionend", () => world.classList.remove("is-gliding"), { once: true });
      }
      applyCam();
    }

    function makeAvatar(person) {
      if (person.avatar) {
        const img = document.createElement("img");
        img.className = "avatar";
        img.src = "/avatar/" + person.avatar;
        img.alt = "";
        return img;
      }
      const span = document.createElement("span");
      span.className = "avatar";
      span.style.background = person.color || "#007aff";
      span.textContent = (person.name || "?").trim().slice(0, 1).toUpperCase();
      return span;
    }

    data.groups.forEach((g) => {
      const n = {
        id: g.id,
        kind: "group",
        data: g,
        x: 0, y: 0, vx: 0, vy: 0,
        r: Math.min(72, 28 + g.name.length * 2.1),
        mass: 4,
        fx: null, fy: null,
      };
      const el = document.createElement("button");
      el.type = "button";
      el.className = "cnode cnode-group";
      el.dataset.id = g.id;
      el.setAttribute("aria-label", g.name);
      const label = document.createElement("span");
      label.className = "cnode-label";
      label.textContent = g.name;
      const count = document.createElement("span");
      count.className = "cnode-n";
      count.textContent = String(g.member_ids.length);
      el.append(label, count);
      n.el = el;
      nodes.push(n);
      byId[g.id] = n;
      nodeLayer.appendChild(el);
    });

    data.people.forEach((p) => {
      const bridge = p.group_ids.length > 2;
      const n = {
        id: p.id,
        kind: "person",
        data: p,
        x: 0, y: 0, vx: 0, vy: 0,
        r: bridge ? 22 : 18,
        mass: 1,
        fx: null, fy: null,
      };
      const el = document.createElement("button");
      el.type = "button";
      el.className = "cnode cnode-person" + (bridge ? " is-bridge" : "");
      el.dataset.id = p.id;
      el.setAttribute("aria-label", p.name);
      el.appendChild(makeAvatar(p));
      const tip = document.createElement("span");
      tip.className = "cnode-tip";
      tip.textContent = p.name;
      el.appendChild(tip);
      n.el = el;
      nodes.push(n);
      byId[p.id] = n;
      nodeLayer.appendChild(el);
    });

    const links = [];
    data.groups.forEach((g) => {
      g.member_ids.forEach((pid) => {
        if (!byId[pid]) return;
        const line = document.createElementNS(SVG_NS, "line");
        line.setAttribute("class", "cedge");
        svg.appendChild(line);
        links.push({ source: byId[pid], target: byId[g.id], el: line, strength: 0.16 });
      });
    });

    const groups = nodes.filter((n) => n.kind === "group");
    const radius = 170 + groups.length * 9;
    groups.forEach((n, i) => {
      const a = (i / Math.max(groups.length, 1)) * Math.PI * 2 - Math.PI / 2;
      n.x = Math.cos(a) * radius;
      n.y = Math.sin(a) * radius;
    });
    nodes.filter((n) => n.kind === "person").forEach((n) => {
      const gs = n.data.group_ids.map((id) => byId[id]).filter(Boolean);
      const len = gs.length || 1;
      n.x = gs.reduce((s, g) => s + g.x, 0) / len + (Math.random() - 0.5) * 28;
      n.y = gs.reduce((s, g) => s + g.y, 0) / len + (Math.random() - 0.5) * 28;
    });

    function place() {
      nodes.forEach((n) => {
        n.el.style.transform = "translate(" + n.x + "px, " + n.y + "px) translate(-50%, -50%)";
      });
      links.forEach((link) => {
        link.el.setAttribute("x1", link.source.x + OFFSET);
        link.el.setAttribute("y1", link.source.y + OFFSET);
        link.el.setAttribute("x2", link.target.x + OFFSET);
        link.el.setAttribute("y2", link.target.y + OFFSET);
      });
    }

    function tick() {
      for (let i = 0; i < links.length; i++) {
        const link = links[i];
        const s = link.source, t = link.target;
        const dx = t.x - s.x, dy = t.y - s.y;
        const dist = Math.hypot(dx, dy) || 1;
        const rest = s.r + t.r + 32;
        const k = ((dist - rest) / dist) * link.strength * alpha;
        const mx = dx * k, my = dy * k;
        s.vx += mx / s.mass;
        s.vy += my / s.mass;
        t.vx -= mx / t.mass;
        t.vy -= my / t.mass;
      }
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const a = nodes[i], b = nodes[j];
          let dx = b.x - a.x, dy = b.y - a.y;
          let dist2 = dx * dx + dy * dy;
          if (dist2 < 0.25) {
            dx = (Math.random() - 0.5) * 0.5;
            dy = (Math.random() - 0.5) * 0.5;
            dist2 = dx * dx + dy * dy;
          }
          const dist = Math.sqrt(dist2);
          const charge = a.kind === "group" && b.kind === "group" ? -520 : -140;
          const force = (charge * alpha) / dist2;
          const fx = (dx / dist) * force;
          const fy = (dy / dist) * force;
          a.vx += fx / a.mass;
          a.vy += fy / a.mass;
          b.vx -= fx / b.mass;
          b.vy -= fy / b.mass;
        }
      }
      for (let i = 0; i < nodes.length; i++) {
        const n = nodes[i];
        n.vx -= n.x * 0.006 * alpha;
        n.vy -= n.y * 0.006 * alpha;
        if (n.fx != null) {
          n.x = n.fx;
          n.y = n.fy;
          n.vx = 0;
          n.vy = 0;
          continue;
        }
        n.vx *= 0.7;
        n.vy *= 0.7;
        n.x += n.vx;
        n.y += n.vy;
      }
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const a = nodes[i], b = nodes[j];
          const dx = b.x - a.x, dy = b.y - a.y;
          const dist = Math.hypot(dx, dy) || 0.01;
          const min = a.r + b.r + 8;
          if (dist >= min) continue;
          const push = (min - dist) / dist * 0.5;
          const ox = dx * push, oy = dy * push;
          if (a.fx == null) { a.x -= ox; a.y -= oy; }
          if (b.fx == null) { b.x += ox; b.y += oy; }
        }
      }
    }

    function heat(value) {
      alpha = Math.max(alpha, value);
      if (!running) {
        running = true;
        loop();
      }
    }

    function loop() {
      tick();
      place();
      alpha *= 0.92;
      if (alpha < 0.02) {
        running = false;
        alpha = 0;
        if (!userMoved) fit(true);
        return;
      }
      requestAnimationFrame(loop);
    }

    function neighborhood(id) {
      const n = byId[id];
      const hot = new Set([id]);
      if (!n) return hot;
      if (n.kind === "person") {
        n.data.group_ids.forEach((gid) => {
          hot.add(gid);
          (byId[gid] && byId[gid].data.member_ids || []).forEach((pid) => hot.add(pid));
        });
      } else {
        n.data.member_ids.forEach((pid) => hot.add(pid));
      }
      return hot;
    }

    function connectedCount(person) {
      const ids = new Set();
      person.group_ids.forEach((gid) => {
        (byId[gid] && byId[gid].data.member_ids || []).forEach((pid) => {
          if (pid !== person.id) ids.add(pid);
        });
      });
      return ids.size;
    }

    function deselect() {
      selected = null;
      world.classList.remove("is-focus");
      nodes.forEach((n) => n.el.classList.remove("is-on", "is-near"));
      links.forEach((link) => link.el.classList.remove("is-hot"));
      sheet.classList.remove("is-open");
    }

    function fillSheet(n) {
      sheet.replaceChildren();
      const head = document.createElement("div");
      head.className = "csheet-head";
      const copy = document.createElement("div");
      const name = document.createElement("div");
      name.className = "csheet-name";
      name.textContent = n.data.name;
      const sub = document.createElement("div");
      sub.className = "csheet-sub";
      copy.append(name, sub);
      if (n.kind === "person") {
        head.append(makeAvatar(n.data), copy);
        const groupsN = n.data.group_ids.length;
        const others = connectedCount(n.data);
        sub.textContent =
          "In " + groupsN + (groupsN === 1 ? " group chat" : " group chats") +
          " · connects " + others + (others === 1 ? " person" : " people");
        const chips = document.createElement("div");
        chips.className = "csheet-chips";
        n.data.group_ids.forEach((gid) => {
          const g = byId[gid];
          if (!g) return;
          const btn = document.createElement("button");
          btn.type = "button";
          btn.className = "csheet-chip";
          btn.textContent = g.data.name;
          btn.addEventListener("click", () => select(gid, true));
          chips.appendChild(btn);
        });
        sheet.append(head, chips);
        if (n.data.chat_id) {
          const open = document.createElement("a");
          open.className = "btn btn-primary csheet-open";
          open.href = "/chat/" + n.data.chat_id;
          open.textContent = "Open chat";
          sheet.appendChild(open);
        }
      } else {
        head.appendChild(copy);
        const members = n.data.member_ids.length;
        sub.textContent =
          members + (members === 1 ? " person" : " people") +
          " · " + n.data.messages.toLocaleString() + " messages";
        const faces = document.createElement("div");
        faces.className = "csheet-people";
        n.data.member_ids.forEach((pid) => {
          const p = byId[pid];
          if (!p) return;
          const btn = document.createElement("button");
          btn.type = "button";
          btn.className = "csheet-face";
          btn.setAttribute("aria-label", p.data.name);
          btn.title = p.data.name;
          btn.appendChild(makeAvatar(p.data));
          btn.addEventListener("click", () => select(pid, true));
          faces.appendChild(btn);
        });
        sheet.append(head, faces);
        if (n.data.chat_id) {
          const open = document.createElement("a");
          open.className = "btn btn-primary csheet-open";
          open.href = "/chat/" + n.data.chat_id;
          open.textContent = "Open group";
          sheet.appendChild(open);
        }
      }
      sheet.classList.add("is-open");
    }

    function select(id, move) {
      const n = byId[id];
      if (!n) return;
      selected = id;
      const hot = neighborhood(id);
      world.classList.add("is-focus");
      nodes.forEach((node) => {
        node.el.classList.toggle("is-on", node.id === id);
        node.el.classList.toggle("is-near", hot.has(node.id) && node.id !== id);
      });
      links.forEach((link) => {
        link.el.classList.toggle("is-hot", hot.has(link.source.id) && hot.has(link.target.id));
      });
      fillSheet(n);
      if (move) panTo(n);
    }

    viewport.addEventListener("pointerdown", (e) => {
      const nodeEl = e.target.closest(".cnode");
      viewport.setPointerCapture(e.pointerId);
      if (nodeEl) {
        const n = byId[nodeEl.dataset.id];
        if (!n) return;
        const w = toWorld(e.clientX, e.clientY);
        drag = { id: n.id, dx: n.x - w.x, dy: n.y - w.y, moved: false, pointerId: e.pointerId };
        select(n.id, false);
        return;
      }
      drag = { pan: true, x: e.clientX, y: e.clientY, moved: false, pointerId: e.pointerId };
      viewport.classList.add("is-panning");
    });

    viewport.addEventListener("pointermove", (e) => {
      if (!drag || drag.pointerId !== e.pointerId) return;
      if (drag.pan) {
        const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
        if (Math.hypot(dx, dy) > 3) drag.moved = true;
        if (drag.moved) {
          userMoved = true;
          cam.x += dx;
          cam.y += dy;
          drag.x = e.clientX;
          drag.y = e.clientY;
          world.classList.remove("is-gliding");
          applyCam();
        }
        return;
      }
      const n = byId[drag.id];
      if (!n) return;
      const w = toWorld(e.clientX, e.clientY);
      const nx = w.x + drag.dx, ny = w.y + drag.dy;
      if (Math.hypot(nx - n.x, ny - n.y) > 2) drag.moved = true;
      if (!drag.moved) return;
      userMoved = true;
      n.fx = nx;
      n.fy = ny;
      heat(0.35);
    });

    function endDrag(e) {
      if (!drag || (e && drag.pointerId !== e.pointerId)) return;
      if (drag.pan && !drag.moved) deselect();
      if (!drag.pan) {
        const n = byId[drag.id];
        if (n) { n.fx = null; n.fy = null; }
        if (drag.moved) heat(0.25);
      }
      drag = null;
      viewport.classList.remove("is-panning");
    }

    viewport.addEventListener("pointerup", endDrag);
    viewport.addEventListener("pointercancel", endDrag);

    viewport.addEventListener("wheel", (e) => {
      e.preventDefault();
      userMoved = true;
      world.classList.remove("is-gliding");
      const rect = viewport.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const wx = (mx - cam.x) / cam.k;
      const wy = (my - cam.y) / cam.k;
      const next = Math.min(2.6, Math.max(0.3, cam.k * (e.deltaY < 0 ? 1.08 : 1 / 1.08)));
      cam.k = next;
      cam.x = mx - wx * cam.k;
      cam.y = my - wy * cam.k;
      applyCam();
    }, { passive: false });

    if (find) {
      find.addEventListener("input", () => {
        const q = find.value.trim().toLowerCase();
        if (!q) return;
        const match = nodes.find((n) => n.data.name.toLowerCase().includes(q));
        if (match) select(match.id, true);
      });
      find.addEventListener("keydown", (e) => {
        if (e.key === "Escape") {
          find.value = "";
          find.blur();
        }
      });
    }

    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && selected && document.activeElement !== find) deselect();
    });

    window.addEventListener("resize", () => {
      if (!userMoved) fit(false);
    });

    const rect = viewport.getBoundingClientRect();
    cam.x = rect.width / 2;
    cam.y = rect.height / 2;
    cam.k = 0.85;
    applyCam();
    place();
    if (reduceMotion) {
      for (let i = 0; i < 220; i++) {
        alpha = 0.08;
        tick();
      }
      alpha = 0;
      place();
      fit(false);
    } else {
      heat(1);
    }
  }

  function initTwinActivity() {
    if (document.getElementById("twinPage")) return;
    async function tick() {
      try {
        const res = await fetch("/twin/status?brief=1");
        applyTwinActivity(await res.json());
      } catch (err) {}
    }
    tick();
    setInterval(tick, 2000);
  }

  initTheme();
  initChatList();
  initDatepicker();
  initHeatMode();
  initHeatmapFocus();
  initChatHeatmap();
  initCountup();
  initTileDurations();
  initMediaSize();
  initMediaRail();
  initChatRail();
  initLightbox();
  initTapbacks();
  initTwin();
  initCircles();
  initTwinActivity();
})();
