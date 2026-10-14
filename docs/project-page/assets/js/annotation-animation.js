(() => {
  "use strict";
  const config = window.VLA_DELTA_SITE?.annotationAnimation;
  const sequence = document.querySelector("[data-annotation-animation]");
  if (!config || !sequence) return;
  const canvas = sequence.querySelector(".annotation-canvas");
  const image = sequence.querySelector("img");
  // Keep the original, accessible static figure in unsupported browsers.
  if (!canvas || !image || typeof canvas.animate !== "function") return;

  let initialized = false;
  const initialize = () => {
    if (initialized || !image.naturalWidth) return;
    const width = image.naturalWidth;
    const height = image.naturalHeight;
    const headerHeight = config.headerHeight;
    const edges = config.panelEdges;
    if (!Array.isArray(edges) || edges.length !== 7 || edges[0] !== 0 || edges[6] !== width ||
        !edges.every((edge, index) => Number.isFinite(edge) && (index === 0 || edge > edges[index - 1])) ||
        !(headerHeight > 0 && headerHeight < height)) return;
    initialized = true;

    const makeLayer = className => {
      const layer = document.createElement("div");
      layer.className = className;
      return layer;
    };
    const background = `url(${JSON.stringify(image.currentSrc || image.src)})`;
    const layers = makeLayer("annotation-layers");
    layers.hidden = true;
    layers.setAttribute("aria-hidden", "true");
    const header = makeLayer("annotation-header");
    header.style.height = `${100 * headerHeight / height}%`;
    header.style.backgroundImage = background;
    const body = makeLayer("annotation-body");
    body.style.top = `${100 * headerHeight / height}%`;
    const panels = edges.slice(0, -1).map((left, index) => {
      const panel = makeLayer("annotation-stage");
      const panelWidth = edges[index + 1] - left;
      panel.style.left = `${100 * left / width}%`;
      panel.style.width = `${100 * panelWidth / width}%`;
      panel.style.backgroundImage = background;
      panel.style.backgroundSize = `${100 * width / panelWidth}% ${100 * height / (height - headerHeight)}%`;
      panel.style.backgroundPosition = `${100 * left / (width - panelWidth)}% 100%`;
      body.append(panel);
      return panel;
    });
    layers.append(header, body);
    canvas.append(layers);

    const duration = (key, fallback) => Number.isFinite(config[key]) && config[key] > 0 ? config[key] : fallback;
    const reveal = duration("revealMs", 900);
    const stagger = duration("staggerMs", 600);
    const lead = 160;
    const total = lead + 5 * stagger + reveal;
    const motion = window.matchMedia("(prefers-reduced-motion: reduce)");
    let started = false;
    let inView = !("IntersectionObserver" in window);
    let animations = [];

    const cancel = () => {
      animations.forEach(animation => { animation.onfinish = null; animation.cancel(); });
      animations = [];
    };
    const updateVisibility = () => {
      animations.forEach(animation => {
        if (inView && !document.hidden) animation.play();
        else animation.pause();
      });
    };
    const showAll = () => {
      cancel();
      started = true;
      layers.hidden = true;
      canvas.classList.remove("is-animating");
    };
    const play = () => {
      cancel();
      started = true;
      layers.hidden = false;
      canvas.classList.add("is-animating");
      const startTime = document.timeline?.currentTime;
      animations = panels.map((panel, index) => {
        const start = lead + index * stagger;
        const animation = panel.animate([
          { opacity: 0, transform: "translateY(10px)", offset: 0 },
          { opacity: 0, transform: "translateY(10px)", offset: start / total, easing: "cubic-bezier(0.22, 1, 0.36, 1)" },
          { opacity: 1, transform: "translateY(0)", offset: (start + reveal) / total },
          { opacity: 1, transform: "translateY(0)", offset: 1 }
        ], { duration: total, iterations: 1, fill: "both" });
        // One shared timeline; replace the completed layers with the original image.
        if (typeof startTime === "number") animation.startTime = startTime;
        return animation;
      });
      animations[animations.length - 1].onfinish = showAll;
      updateVisibility();
    };

    motion.addEventListener("change", () => {
      if (motion.matches) {
        showAll();
      }
    });
    document.addEventListener("visibilitychange", updateVisibility);
    if ("IntersectionObserver" in window) {
      const observer = new window.IntersectionObserver(entries => {
        inView = entries.some(entry => entry.isIntersecting);
        if (inView && !started && !motion.matches) play();
        updateVisibility();
      }, { threshold: 0.25 });
      observer.observe(sequence);
    } else if (!motion.matches) play();
  };

  image.addEventListener("load", initialize, { once: true });
  if (image.getAttribute("src") !== config.poster) image.src = config.poster;
  sequence.querySelectorAll("[data-animation-full-size]").forEach(link => { link.href = config.poster; });
  if (image.complete) initialize();
})();
