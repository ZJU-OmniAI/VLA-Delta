(() => {
  "use strict";
  const figure = document.querySelector("[data-pipeline-animation]");
  const config = window.VLA_DELTA_SITE?.pipelineAnimation;
  if (!figure || !config || typeof figure.animate !== "function") return;
  const svg = figure.querySelector(".pipeline-diagram");
  const stages = Array.from(svg.querySelectorAll("[data-pipeline-step]"));
  const routes = Array.from(svg.querySelectorAll("[data-pipeline-to]"));
  const feedback = svg.querySelector(".pipeline-feedback");
  const status = figure.querySelector("[data-pipeline-status]");
  if (stages.length !== 8 || routes.length !== 7 || !feedback || !status) return;

  const setting = (key, fallback) => Number.isFinite(config[key]) && config[key] > 0 ? config[key] : fallback;
  const reveal = setting("revealMs", 850);
  const step = Math.max(setting("stepMs", 1000), reveal);
  const review = setting("feedbackMs", 1600);
  const lead = 180;
  const feedbackStart = lead + 6 * step + reveal + 180;
  const finalStart = feedbackStart + review + 240;
  const starts = Array.from({ length: 8 }, (_, i) => i < 7 ? lead + i * step : finalStart);
  const duration = finalStart + reveal + 200;
  const labels = ["Data selection", "Unified format", "Data cleaning", "Keyframes + semantics", "Visual annotation", "Sample construction", "LLM review", "Human review"];
  const motion = matchMedia("(prefers-reduced-motion: reduce)");
  let inView = !("IntersectionObserver" in window);
  let playing = false;
  let started = false;
  let animations = [];
  let ticker = 0;
  let lastLabel = "";

  const writeStatus = text => {
    if (text !== lastLabel) { status.textContent = text; lastLabel = text; }
  };
  const clear = () => {
    cancelAnimationFrame(ticker);
    animations.forEach(animation => { animation.onfinish = null; animation.cancel(); });
    animations = [];
  };
  const complete = () => {
    playing = false;
    clear();
    figure.dataset.state = "complete";
    writeStatus("Complete pipeline · 8 steps");
  };
  const tick = () => {
    if (!playing || !inView || document.hidden) return;
    const time = Number(animations[0]?.currentTime || 0);
    if (time >= feedbackStart && time < finalStart) {
      writeStatus("Review feedback · Revise prompts and rules");
    } else {
      const index = Math.max(0, starts.findLastIndex(start => time >= start));
      writeStatus(`${index + 1} / 8 · ${labels[index]}`);
    }
    ticker = requestAnimationFrame(tick);
  };
  const visibility = () => {
    cancelAnimationFrame(ticker);
    if (!playing) return;
    animations.forEach(animation => inView && !document.hidden ? animation.play() : animation.pause());
    if (inView && !document.hidden) tick();
  };
  const add = (element, keyframes) => {
    const animation = element.animate(keyframes, { duration, fill: "both" });
    animation.pause();
    animation.currentTime = 0;
    animations.push(animation);
  };
  const fade = (element, start, length, initial = 0) => add(element, [
    { opacity: initial, offset: 0 },
    { opacity: initial, offset: Math.max(0, start) / duration, easing: "cubic-bezier(.22,1,.36,1)" },
    { opacity: 1, offset: (start + length) / duration },
    { opacity: 1, offset: 1 }
  ]);
  const play = () => {
    clear();
    started = true;
    playing = true;
    figure.dataset.state = "playing";
    stages.forEach((stage, index) => fade(stage, starts[index], reveal, .12));
    routes.forEach(route => {
      const target = Number(route.dataset.pipelineTo);
      const start = starts[target - 1] - (target === 5 ? 700 : 300);
      if (target === 5) {
        // The original 4→5 connector turns down, left, then down to the second row.
        Array.from(route.children).forEach((part, index) => {
          const begin = start + index * 150;
          fade(part, begin, 220);
          const path = part.querySelector("path");
          if (path) {
            const length = path.getTotalLength();
            add(path, [
              { strokeDasharray: `${length} ${length}`, strokeDashoffset: length, offset: 0 },
              { strokeDasharray: `${length} ${length}`, strokeDashoffset: length, offset: begin / duration },
              { strokeDasharray: `${length} ${length}`, strokeDashoffset: 0, offset: (begin + 260) / duration },
              { strokeDasharray: `${length} ${length}`, strokeDashoffset: 0, offset: 1 }
            ]);
          }
        });
      } else fade(route, start, 400);
    });
    // Keep the original dashed feedback path. Reveal segments from step 7 back to 4.
    Array.from(feedback.children).forEach((part, index) => {
      const isLabel = part.dataset.sourceName.endsWith("label");
      fade(part, feedbackStart + (isLabel ? 0 : index * review / 6), isLabel ? 700 : 450);
    });
    animations[0].onfinish = complete;
    visibility();
  };

  figure.querySelector(".pipeline-toolbar").hidden = false;
  motion.addEventListener("change", () => { if (motion.matches) { started = true; complete(); } });
  document.addEventListener("visibilitychange", visibility);
  if ("IntersectionObserver" in window) {
    const observer = new IntersectionObserver(entries => {
      inView = entries.some(entry => entry.isIntersecting);
      if (inView && !started && !motion.matches) play();
      else visibility();
    }, { threshold: .15 });
    observer.observe(svg);
  } else if (!motion.matches) play();
})();
