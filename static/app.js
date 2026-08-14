(function () {
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const THEME_KEY = "theme";

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

  initTheme();
  initChatList();
  initDatepicker();
  initHeatmapFocus();
  initCountup();
  initTileDurations();
  initMediaSize();
  initMediaRail();
  initChatRail();
  initLightbox();
})();
