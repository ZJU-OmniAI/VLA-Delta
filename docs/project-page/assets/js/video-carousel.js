(() => {
  "use strict";
  const track = document.querySelector("[data-robot-videos]");
  if (!track) return;
  const cards = Array.from(track.querySelectorAll(".video-card"));
  const videos = cards.map(card => card.querySelector("video")).filter(Boolean);
  if (!cards.length) return;

  const settings = window.VLA_DELTA_SITE?.videoCarousel || {};
  const speed = Math.max(0, Number(settings.speedPxPerSecond) || 28);
  const hoverScale = Math.max(1, Math.min(1.12, Number(settings.hoverScale) || 1.05));
  const holdMs = Math.max(0, Number(settings.interactionPauseMs) || 1400);
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const canHover = window.matchMedia("(hover: hover) and (pointer: fine)");
  const hovered = new Set();
  let inView = false;
  let active = false;
  let keyboardPaused = false;
  let holdUntil = 0;
  let frame = 0;
  let lastTime = 0;
  let offset = 0;
  let stepSize = 0;
  let cardWidth = 0;
  let viewWidth = 0;
  let cycle = 0;
  let tween = null;
  let drag = null;
  let dragged = false;
  const mod = (value, period) => ((value % period) + period) % period;

  // Five original video nodes stay mounted. Only their positions wrap outside the
  // visible area; no duplicated players, source reloads or visible rewind at a seam.
  track.classList.add("is-looping");
  track.style.setProperty("--video-hover-scale", String(hoverScale));
  const render = () => {
    if (!cycle) return;
    cards.forEach((card, index) => {
      const x = mod(index * stepSize + offset + stepSize, cycle) - stepSize;
      card.style.translate = `${x - index * stepSize}px 0`;
      card.style.transformOrigin = x < 12 ? "left center" :
        x + cardWidth > viewWidth - 12 ? "right center" : "center";
    });
  };
  const measure = () => {
    const oldStep = stepSize;
    const gap = parseFloat(getComputedStyle(track).columnGap) || 0;
    cardWidth = cards[0].offsetWidth;
    viewWidth = track.clientWidth;
    stepSize = cardWidth + gap;
    cycle = stepSize * cards.length;
    if (oldStep) offset = offset / oldStep * stepSize;
    offset = mod(offset, cycle);
    tween = null;
    track.scrollLeft = 0;
    render();
  };
  const playVideos = () => {
    if (reducedMotion.matches) return;
    videos.forEach(video => {
      if (video.hidden) return;
      const attempt = video.play();
      if (attempt) attempt.catch(() => { /* Native play controls remain available. */ });
    });
  };
  const animate = now => {
    frame = 0;
    if (!active) return;
    const seconds = lastTime ? Math.min((now - lastTime) / 1000, 0.05) : 0;
    lastTime = now;
    const paused = reducedMotion.matches || hovered.size > 0 ||
      keyboardPaused || drag !== null || now < holdUntil;
    if (tween) {
      const progress = Math.min(1, (now - tween.start) / tween.duration);
      const ease = 1 - Math.pow(1 - progress, 3);
      offset = tween.from + (tween.to - tween.from) * ease;
      render();
      if (progress === 1) { offset = mod(offset, cycle); tween = null; }
    } else if (!paused && cards.length > 1) {
      offset = mod(offset + speed * seconds, cycle); // Cards travel to the right.
      render();
    }
    track.dataset.motion = paused ? "paused" : "running";
    frame = requestAnimationFrame(animate);
  };
  const setActive = () => {
    const nextActive = inView && !document.hidden;
    if (nextActive === active) return;
    active = nextActive;
    lastTime = 0;
    if (active) {
      playVideos();
      frame = requestAnimationFrame(animate);
    } else {
      cancelAnimationFrame(frame);
      frame = 0;
      tween = null;
      videos.forEach(video => video.pause());
      track.dataset.motion = "offscreen";
    }
  };
  const hold = () => { holdUntil = performance.now() + holdMs; };
  const moveBy = amount => {
    hold();
    if (reducedMotion.matches || !active) {
      offset = mod(offset + amount, cycle);
      tween = null;
      render();
    } else tween = { from: offset, to: offset + amount, start: performance.now(), duration: 350 };
  };
  cards.forEach(card => {
    card.addEventListener("pointerenter", event => {
      if (!canHover.matches || event.pointerType !== "mouse") return;
      hovered.add(card);
      card.classList.add("is-hovered");
      tween = null;
    });
    card.addEventListener("pointerleave", () => {
      hovered.delete(card);
      card.classList.remove("is-hovered");
    });
  });
  track.addEventListener("focusin", event => {
    keyboardPaused = event.target.matches(":focus-visible");
    const card = event.target.closest(".video-card");
    if (card && keyboardPaused) {
      offset = mod(-cards.indexOf(card) * stepSize, cycle);
      tween = null;
      render();
    }
    track.scrollLeft = 0;
  });
  track.addEventListener("focusout", event => {
    if (!track.contains(event.relatedTarget)) keyboardPaused = false;
  });
  track.addEventListener("keydown", event => {
    if (event.target !== track || event.altKey || event.ctrlKey || event.metaKey) return;
    keyboardPaused = true;
    if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
      event.preventDefault();
      moveBy(event.key === "ArrowLeft" ? stepSize : -stepSize);
    } else if (event.key === "Home" || event.key === "End") {
      event.preventDefault();
      const target = event.key === "Home" ? 0 : -stepSize * (cards.length - 1);
      moveBy(target - offset);
    }
  });
  track.addEventListener("wheel", event => {
    if (event.ctrlKey || cards.length < 2) return;
    // Keep vertical page scrolling available even though this row has no endpoint.
    if (!event.shiftKey && Math.abs(event.deltaY) >= Math.abs(event.deltaX)) return;
    const unit = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? track.clientWidth : 1;
    const delta = (Math.abs(event.deltaX) > Math.abs(event.deltaY) ? event.deltaX : event.deltaY) * unit;
    if (!delta) return;
    event.preventDefault();
    tween = null;
    offset = mod(offset - delta, cycle);
    hold();
    render();
  }, { passive: false });

  // Mouse dragging uses captions/card padding. Touch swipes also work over the video
  // picture; its bottom control strip is reserved for seeking and native controls.
  track.addEventListener("pointerdown", event => {
    dragged = false;
    if (!event.isPrimary || event.button !== 0 || event.target.closest("a, button, input")) return;
    const video = event.target.closest("video");
    if (video && (event.pointerType === "mouse" ||
        event.clientY > video.getBoundingClientRect().bottom - 48)) return;
    drag = { id: event.pointerId, x: event.clientX, y: event.clientY, offset };
    tween = null;
    hold();
    if (event.pointerType === "mouse") event.preventDefault();
  });
  track.addEventListener("pointermove", event => {
    if (!drag || event.pointerId !== drag.id) return;
    const dx = event.clientX - drag.x;
    const dy = event.clientY - drag.y;
    if (!dragged && Math.abs(dy) > Math.abs(dx) && Math.abs(dy) > 8) { drag = null; return; }
    if (!dragged && Math.abs(dx) < 5) return;
    if (!dragged) {
      dragged = true;
      track.classList.add("is-dragging");
      track.setPointerCapture(event.pointerId);
    }
    offset = mod(drag.offset + dx, cycle);
    render();
  });
  const finishDrag = event => {
    if (!drag || event.pointerId !== drag.id) return;
    drag = null;
    track.classList.remove("is-dragging");
    if (track.hasPointerCapture(event.pointerId)) track.releasePointerCapture(event.pointerId);
    hold();
  };
  track.addEventListener("pointerup", finishDrag);
  track.addEventListener("pointercancel", finishDrag);
  track.addEventListener("lostpointercapture", finishDrag);
  track.addEventListener("pointerleave", event => { if (!dragged) finishDrag(event); });
  track.addEventListener("click", event => {
    if (!dragged) return;
    event.preventDefault();
    event.stopPropagation();
    dragged = false;
  }, true);

  measure();
  if ("ResizeObserver" in window) new ResizeObserver(measure).observe(track);
  else window.addEventListener("resize", measure);
  if ("IntersectionObserver" in window) {
    new IntersectionObserver(entries => {
      inView = entries[0].isIntersecting && entries[0].intersectionRatio >= 0.2;
      setActive();
    }, { threshold: [0, 0.2], rootMargin: "-90px 0px 0px 0px" }).observe(track);
  } else { inView = true; setActive(); }
  document.addEventListener("visibilitychange", setActive);
  reducedMotion.addEventListener("change", () => {
    if (reducedMotion.matches) videos.forEach(video => video.pause());
    else if (active) playVideos();
  });
})();
