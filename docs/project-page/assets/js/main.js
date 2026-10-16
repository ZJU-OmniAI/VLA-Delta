(() => {
  "use strict";
  const config = window.VLA_DELTA_SITE;
  if (!config) return; // The static HTML remains readable without configuration.

  const make = (tag, text, className) => {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    if (className) node.className = className;
    return node;
  };
  const available = (href) => {
    if (typeof href !== "string" || !href.trim() || href.trim() === "#") return false;
    try {
      return ["http:", "https:", "file:"].includes(new URL(href, document.baseURI).protocol);
    } catch { return false; }
  };

  if (config.logo && available(config.logo.src)) {
    document.querySelectorAll("[data-project-logo]").forEach(logo => {
      logo.src = config.logo.src;
      logo.width = config.logo.width;
      logo.height = config.logo.height;
      logo.alt = config.logo.alt;
    });
  }
  document.title = `${config.title} | Project Page`;
  const title = document.querySelector("[data-project-title]");
  if (title) {
    const colon = config.title.indexOf(":");
    if (colon >= 0) {
      const project = make("span", undefined, "title-project");
      const logo = title.querySelector("[data-project-logo]");
      if (logo) {
        project.append(logo);
      }
      project.append(document.createTextNode(config.title.slice(0, colon + 1)));
      title.replaceChildren(project,
        document.createTextNode(` ${config.title.slice(colon + 1).trim()}`));
    } else title.textContent = config.title;
  }
  const tagline = document.querySelector("[data-tagline]");
  if (tagline) tagline.textContent = config.tagline;
  const authors = document.querySelector("[data-authors]");
  if (authors) authors.replaceChildren(...config.authors.map(author =>
    make("span", `${author.given} ${author.family}`, "author")
  ));
  const affiliations = document.querySelector("[data-affiliations]");
  if (affiliations) affiliations.replaceChildren(...config.affiliations.map(affiliation => {
    const node = make("span");
    const name = make(available(affiliation.url) ? "a" : "span", affiliation.name);
    if (name.tagName === "A") {
      name.href = affiliation.url;
      name.target = "_blank";
      name.rel = "noopener noreferrer";
    }
    node.append(name, document.createTextNode(" "),
      make("span", `(${affiliation.domain})`, "domain"));
    return node;
  }));

  document.querySelectorAll("[data-resource]").forEach(original => {
    const key = original.dataset.resource;
    const href = config.resources[key];
    const ready = available(href);
    const labels = { github: "GitHub", huggingface: "Hugging Face", paper: "Paper", project: "Project Page", arxiv: "arXiv" };
    const node = make(ready ? "a" : "span", undefined,
      `resource${key === "github" && ready ? " primary" : ""}${ready ? "" : " unavailable"}`);
    node.dataset.resource = key;
    const icon = original.querySelector("svg");
    if (icon) node.append(icon.cloneNode(true));
    node.append(make("span", ready ? labels[key] : `${labels[key]} — Coming Soon`));
    if (ready) {
      node.href = href;
      node.target = "_blank";
      node.rel = "noopener noreferrer";
    } else node.setAttribute("aria-disabled", "true");
    original.replaceWith(node);
  });
  document.querySelectorAll("[data-paper-link]").forEach(link => {
    if (available(config.resources.paper)) link.href = config.resources.paper;
    else link.replaceWith(make("span", link.textContent));
  });

  document.querySelectorAll("[data-figure]").forEach(figure => {
    const asset = config.figures[figure.dataset.figure];
    if (!asset) return;
    const image = figure.querySelector("img");
    const source = figure.querySelector("source");
    if (image) {
      image.src = `${asset.base}.png`;
      image.width = asset.width;
      image.height = asset.height;
    }
    if (source) source.srcset = asset.smallWidth < asset.width
      ? `${asset.base}-1200.webp ${asset.smallWidth}w, ${asset.base}.webp ${asset.width}w`
      : `${asset.base}.webp`;
    figure.querySelectorAll("[data-full-size]").forEach(link => {
      link.href = `${asset.base}.png`;
    });
  });
  document.querySelectorAll("[data-full-figure]").forEach(link => {
    const asset = config.figures[link.dataset.fullFigure];
    if (asset) link.href = `${asset.base}.png`;
  });

  const bibtex = document.querySelector("#bibtex");
  if (bibtex) {
    const fields = {
      title: config.title.replaceAll("Δ", "{$\\Delta$}"),
      author: config.authors.map(author => `${author.family}, ${author.given}`).join(" and "),
      ...config.citation.fields
    };
    const entries = Object.entries(fields).filter(([, value]) => value !== null && value !== undefined && value !== "");
    bibtex.textContent = `@${config.citation.entryType}{${config.citation.key},\n` +
      entries.map(([key, value]) => `  ${key} = {${value}}`).join(",\n") + "\n}";
  }
  const citationStatus = document.querySelector("[data-citation-status]");
  if (citationStatus) citationStatus.textContent = config.citation.provisional
    ? "Provisional citation · publication metadata pending"
    : "BibTeX citation";
  const citationNote = document.querySelector("[data-citation-note]");
  if (citationNote) citationNote.textContent = config.citation.provisional
    ? "Title and authors are confirmed. Year, publication venue and public identifier will be added when available."
    : "Please use the following citation when referring to this work.";

  const copyButton = document.querySelector("[data-copy-bibtex]");
  const copyStatus = document.querySelector("#copy-status");
  let resetCopyLabel;
  const selectBibtex = () => {
    const selection = window.getSelection();
    if (!selection) return;
    const range = document.createRange();
    range.selectNodeContents(bibtex);
    selection.removeAllRanges();
    selection.addRange(range);
    bibtex.parentElement.focus({ preventScroll: true });
  };
  if (copyButton && bibtex && copyStatus) {
    copyButton.hidden = false;
    copyButton.addEventListener("click", async () => {
      clearTimeout(resetCopyLabel);
      copyButton.disabled = true;
      const label = copyButton.querySelector("span");
      label.textContent = "Copying…";
      let copied = false;
      try {
        if (navigator.clipboard && navigator.clipboard.writeText) {
          await navigator.clipboard.writeText(bibtex.textContent);
          copied = true;
        }
      } catch { /* Try the legacy path, then offer manual selection. */ }
      if (!copied) {
        selectBibtex();
        try { copied = typeof document.execCommand === "function" && document.execCommand("copy"); }
        catch { copied = false; }
      }
      copyButton.disabled = false;
      if (copied) {
        label.textContent = "Copied!";
        copyStatus.textContent = "BibTeX copied to the clipboard.";
        resetCopyLabel = setTimeout(() => { label.textContent = "Copy BibTeX"; }, 2200);
      } else {
        label.textContent = "Copy BibTeX";
        copyStatus.textContent = "Automatic copying is unavailable. The BibTeX is selected; press Ctrl+C (⌘C on Mac) to copy.";
      }
    });
  }

  // The narrated overview starts only when the visitor presses play.
  // Static HTML supplies the same player when JavaScript is unavailable.
  const heroVideo = document.querySelector("[data-hero-video]");
  const featured = config.heroVideo;
  if (heroVideo && featured && available(featured.src)) {
    const video = heroVideo.querySelector("[data-hero-player]") || make("video");
    video.setAttribute("data-hero-player", "");
    video.controls = true;
    video.playsInline = true;
    video.preload = "none";
    video.width = featured.width || 1920;
    video.height = featured.height || 1080;
    if (available(featured.poster)) video.poster = featured.poster;
    else video.removeAttribute("poster");
    video.setAttribute("aria-label", featured.title || "VLA-Δ project video");
    video.addEventListener("error", () => {
      heroVideo.replaceChildren(make("p", "Video currently unavailable", "hero-video-caption"));
    }, { once: true });
    if (video.getAttribute("src") !== featured.src) video.src = featured.src;
    const fallback = make("a", "Open the VLA-Δ introduction video");
    fallback.href = featured.src;
    video.replaceChildren(fallback);
    heroVideo.replaceChildren(video);
    if (featured.description) heroVideo.append(make("p", featured.description, "hero-video-caption"));
  } else if (heroVideo) {
    const placeholder = make("div", undefined, "video-placeholder");
    placeholder.setAttribute("role", "status");
    placeholder.append(make("p", "Video — Coming Soon"));
    heroVideo.replaceChildren(placeholder);
  }

  // Keep the five clips inside Real-Robot Evaluation. Static HTML is the no-JS fallback.
  const demoBlock = document.querySelector(".real-demos");
  const grid = document.querySelector("[data-robot-videos]");
  const videos = (config.videos || []).filter(video => available(video.src) && video.title);
  if (demoBlock && grid) {
    demoBlock.hidden = videos.length === 0;
    grid.replaceChildren(...videos.map(data => {
      const figure = make("figure", undefined, "video-card");
      const video = make("video");
      video.controls = true;
      video.muted = true;
      video.loop = true;
      video.playsInline = true;
      video.preload = "none";
      video.width = data.width || 1280;
      video.height = data.height || 720;
      video.src = data.src;
      if (available(data.poster)) video.poster = data.poster;
      video.setAttribute("aria-label", data.title);
      const download = make("a", "Open MP4 video");
      download.href = data.src;
      video.append(download);
      const caption = make("figcaption");
      caption.append(make("strong", data.title));
      if (data.description) caption.append(make("p", data.description));
      video.addEventListener("error", () => {
        video.hidden = true;
        const notice = make("p", "This clip is currently unavailable. ");
        notice.append(download.cloneNode(true));
        caption.append(notice);
      }, { once: true });
      figure.append(video, caption);
      return figure;
    }));
  }
})();
