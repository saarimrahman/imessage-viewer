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
    const sendBtn = document.getElementById("twinSend");
    const input = document.getElementById("twinInput");
    const thread = document.getElementById("twinThread");
    const form = document.getElementById("twinCompose");
    const chatModelEl = document.getElementById("twinChatModel");
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
    const WAITING_PHASES = ["inspecting", "exporting", "downloading", "loading"];
    const history = [];
    let pollTimer = null;
    let sending = false;
    let lastStatus = null;
    let wasBusy = page.classList.contains("is-busy");

    function selectedModel() {
      return modelSelect.value || "qwen3-capable";
    }

    function selectedModelInfo(s) {
      return (s.models || []).find((model) => model.key === selectedModel());
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

    function emptyState() {
      if (thread.querySelector(".row")) return;
      if (thread.querySelector(".twin-empty")) return;
      const p = document.createElement("p");
      p.className = "twin-empty";
      p.append("Train a model, then text it here. You are on the right; your twin replies on the left. ");
      const a = document.createElement("a");
      a.href = "#model";
      a.textContent = "Choose a model";
      p.append(a);
      thread.appendChild(p);
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
      return row;
    }

    function svgNode(name, attrs, text) {
      const node = document.createElementNS("http://www.w3.org/2000/svg", name);
      Object.keys(attrs || {}).forEach((key) => node.setAttribute(key, attrs[key]));
      if (text !== undefined) node.textContent = text;
      return node;
    }

    function chartPath(points, key, width, height, maxX, maxY) {
      const filtered = points.filter((point) => Number.isFinite(point[key]));
      return filtered.map((point, index) => {
        const x = 34 + (point.iter / Math.max(1, maxX)) * (width - 46);
        const y = 10 + (1 - point[key] / Math.max(0.001, maxY)) * (height - 36);
        return (index ? "L" : "M") + x.toFixed(1) + " " + y.toFixed(1);
      }).join(" ");
    }

    function drawChart(svg, metrics, keys, maxY) {
      const width = 520;
      const height = 190;
      const maxX = Math.max(1, ...metrics.map((point) => point.iter || 0));
      const values = metrics.flatMap((point) => keys.map((key) => point[key])).filter(Number.isFinite);
      const yTop = maxY || Math.max(1, ...values) * 1.12;
      const grid = svg.querySelector(".chart-grid");
      const labels = svg.querySelector(".chart-labels");
      grid.replaceChildren();
      labels.replaceChildren();
      for (let i = 0; i <= 3; i += 1) {
        const y = 10 + (i / 3) * (height - 36);
        grid.appendChild(svgNode("line", { x1: 34, y1: y, x2: width - 12, y2: y }));
        labels.appendChild(svgNode("text", { x: 30, y: y + 3, "text-anchor": "end" }, (yTop * (1 - i / 3)).toFixed(yTop < 10 ? 1 : 0)));
      }
      labels.appendChild(svgNode("text", { x: 34, y: height - 4 }, "0"));
      labels.appendChild(svgNode("text", { x: width - 12, y: height - 4, "text-anchor": "end" }, maxX.toLocaleString()));
      keys.forEach((key, index) => {
        const path = svg.querySelectorAll(".chart-line")[index];
        if (path) path.setAttribute("d", chartPath(metrics, key, width, height, maxX, yTop));
      });
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
    }

    const STEP_COPY = [
      "Count usable text without exposing it.",
      "Keep context and derive short real variants.",
      "Download weights if needed, then train on your JSONL.",
      "Opens automatically when the adapter is ready.",
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
          small.textContent = (index === active && STEP_LIVE[phase]) ? STEP_LIVE[phase] : STEP_COPY[index];
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
      (s.models || []).forEach((model) => {
        const option = Array.from(modelSelect.options).find((item) => item.value === model.key);
        if (!option) return;
        option.textContent = model.name + " — " + model.params + " · " + model.download + (model.has_adapter ? " · trained" : "");
      });
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
      if (!live || row.status !== "running") return row;
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

    function openRunProgress() {
      showTab("model", true);
      setHash("model");
      const metrics = document.getElementById("twinMetrics");
      const live = document.getElementById("twinLive");
      const hasChart = metrics && !metrics.hidden && ((lastStatus && lastStatus.metrics) || []).length > 0;
      const target = hasChart ? metrics : live;
      if (!target) return;
      document.querySelectorAll(".twin-metrics.is-revealed, .twin-live.is-revealed").forEach((el) => {
        el.classList.remove("is-revealed");
      });
      requestAnimationFrame(() => {
        target.scrollIntoView({ block: "start", behavior: reduceMotion ? "auto" : "smooth" });
        target.classList.add("is-revealed");
        window.setTimeout(() => target.classList.remove("is-revealed"), reduceMotion ? 0 : 1200);
      });
    }

    function renderRuns(runs, live) {
      const box = document.getElementById("twinRuns");
      const list = document.getElementById("twinRunList");
      if (!box || !list) return;
      const rows = runs || [];
      box.hidden = rows.length === 0;
      list.replaceChildren();
      rows.forEach((raw) => {
        const row = liveRun(raw, live);
        const status = row.status === "ready" ? "Done"
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
        kindLabel.textContent = row.run === "quick" ? "Quick" : "Complete";
        sub.append(kindLabel);
        if (row.params) {
          const size = document.createElement("span");
          size.textContent = row.params;
          sub.append(size);
        }
        if (kind === "running") {
          const open = document.createElement("a");
          open.className = "twin-run-open";
          open.href = "#twinLive";
          open.textContent = "View progress";
          open.setAttribute("aria-label", "View progress for " + (row.name || "this run"));
          sub.append(open);
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
        if (row.status !== "ready" && row.iters) {
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
      sendBtn.disabled = busy || !modelReady;
      input.disabled = busy || !modelReady;
      modelSelect.disabled = busy;
      syncModelPicker(s);
      updateModelSummary(model);
      chatModelEl.textContent = model
        ? model.name + " · " + model.params + (modelReady ? " · ready" : " · not trained")
        : "Select a trained model in Model.";
      if (!s.mlx) {
        statusEl.textContent = "mlx-lm is not installed.";
      } else if (WAITING_PHASES.includes(s.phase) || s.phase === "training" || s.phase === "cancelling") {
        statusEl.textContent = s.detail;
      } else if (s.phase === "error") {
        statusEl.textContent = s.detail || "Training failed.";
      } else if (s.phase === "cancelled") {
        statusEl.textContent = "Training stopped.";
      } else if (modelReady) {
        statusEl.textContent = model.name + " adapter ready. Send a text or retrain it.";
      } else {
        statusEl.textContent = model ? model.name + " has not been trained yet." : "Choose a model.";
      }
      const losses = latestLosses(s.metrics);
      const bits = [];
      if (s.phase === "training" && losses.train != null) bits.push("train " + formatLoss(losses.train));
      if (s.phase === "training" && losses.reference != null) bits.push("ref " + formatLoss(losses.reference));
      const eta = formatDuration(s.eta_seconds);
      if (eta) bits.push("~" + eta + " left");
      else if (busy && s.elapsed_seconds) bits.push(formatDuration(s.elapsed_seconds));
      else if (busy) bits.push("Step " + (s.phase === "inspecting" ? 1 : s.phase === "exporting" ? 2 : 3) + " of 4");
      else if (s.phase === "cancelled") bits.push("Stopped");
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
        showTab("chat", true);
        setHash("chat");
      }
      wasBusy = busy;
      if (hintEl) {
        if (!s.mlx) {
          hintEl.innerHTML =
            "Install MLX training support in this app's virtualenv, then restart the server:" +
            "<pre>./.venv/bin/python -m pip install -r twin/requirements.txt</pre>";
        } else if (s.phase === "error") {
          hintEl.textContent = "The previous run stopped. Quick is the shortest way to verify the setup before retrying Complete.";
        } else if (s.phase === "cancelled") {
          hintEl.textContent = "Training was stopped. Start again when you want to fit this model.";
        } else if (busy && s.examples) {
          hintEl.textContent = s.examples.toLocaleString() + " examples cover " + s.sent_texts.toLocaleString() + " sent texts across " + s.chats.toLocaleString() + " chats" + (s.augmented ? ", including " + s.augmented.toLocaleString() + " short-context variants." : ".");
        } else if (modelReady) {
          hintEl.textContent = "This adapter stays paired with " + model.name + ". Training another size creates a separate adapter.";
        } else {
          hintEl.textContent = "Quick uses a small recent slice. Complete uses every chat and makes one pass through all generated examples.";
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

    async function loadDataProfile() {
      const note = document.getElementById("twinDataNote");
      try {
        const res = await fetch("/twin/data");
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

    stopBtn.addEventListener("click", async () => {
      stopBtn.disabled = true;
      try {
        const res = await fetch("/twin/stop", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: "{}",
        });
        applyStatus(await res.json());
      } catch (err) {
        stopBtn.disabled = false;
        statusEl.textContent = "Could not stop training.";
      }
    });

    trainBtn.addEventListener("click", async () => {
      const run = (page.querySelector('input[name="twinrun"]:checked') || {}).value || "complete";
      const model = selectedModel();
      trainBtn.disabled = true;
      revealSignals(true);
      showTab("model", true);
      setHash("model");
      try {
        const res = await fetch("/twin/train", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ run, model }),
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
      history.length = 0;
      thread.replaceChildren();
      emptyState();
      if (lastStatus) applyStatus(lastStatus);
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
      try {
        const res = await fetch("/twin/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text, history, model: selectedModel() }),
        });
        const data = await res.json();
        const bubble = pending.querySelector(".bubble");
        if (!res.ok) {
          bubble.textContent = data.error || "The twin could not reply.";
        } else {
          bubble.textContent = data.reply;
          history.push({ role: "user", content: text });
          history.push({ role: "assistant", content: data.reply });
        }
        pending.removeAttribute("data-pending");
      } catch (err) {
        pending.querySelector(".bubble").textContent = "The twin could not reply.";
      }
      sending = false;
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
      if (!e.target.closest(".twin-run.is-openable")) return;
      e.preventDefault();
      openRunProgress();
    });
    loadDataProfile();
    refreshStatus();
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
  initCountup();
  initTileDurations();
  initMediaSize();
  initMediaRail();
  initChatRail();
  initLightbox();
  initTapbacks();
  initTwin();
  initTwinActivity();
})();
