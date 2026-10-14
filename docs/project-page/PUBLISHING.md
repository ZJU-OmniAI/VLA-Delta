# Publishing the VLA-Delta project page

Public URL: https://zju-omniai.github.io/VLA-Delta/

The editable website is in `docs/project-page/`. GitHub Actions publishes this
directory as the website root, so the public URL does not contain `docs/` or
`project-page/`. The site needs no backend, npm installation, or model environment.

## Files and updates

- `index.html`: page content and static fallbacks.
- `assets/js/site-config.js`: title, authors, links, figures, and video settings.
- `assets/css/style.css`: layout and styling.
- `publish-files.txt`: the exact list of files included in the public website.
- `.gitignore`: excludes local authoring material and allows the release files.
- `../../.github/workflows/pages.yml`: validates and publishes updates to `main`.

The release includes the introduction video, conceptual animation, and five
robot demonstrations. Video production environments, model caches, source PPTs,
unlinked manuscript PDF, intermediate exports, and local review notes stay local.
They are not needed to view the website. Their original files are not deleted.

Run from the repository root:

```bash
python3 docs/project-page/scripts/check-site.py
python3 -m http.server 8001 --directory docs/project-page
```

Open http://localhost:8001/ to preview. Update both `site-config.js` and the
matching HTML fallback when changing page content. For a new asset, add its
relative path to `publish-files.txt` and allow it in this directory's `.gitignore`.
Confirm that it is staged, particularly for videos, which the root ignore rules
normally exclude. Commit and push the website changes to `main` to publish.

In GitHub, **Settings → Pages → Source** must be **GitHub Actions**. The
**Deploy project page** workflow validates the release files, uploads only the
manifest entries, and deploys the artifact. It can also be run manually from
the Actions tab. Local scripts and this guide are versioned but not served.

## Custom domain

The default GitHub Pages address requires no domain purchase. To use a domain
you own, such as `vla-delta.example.com`:

1. Set **Settings → Pages → Custom domain** to `vla-delta.example.com` and save.
2. In your DNS provider, add a `CNAME` record for `vla-delta` pointing to
   `zju-omniai.github.io` (without a protocol or repository path).
3. Wait for the DNS check and HTTPS certificate, then enable **Enforce HTTPS**.
4. Update the project-page link in the root README and the URL in this guide.

For an apex domain such as `example.com`, follow GitHub's A/ALIAS/ANAME setup.
This deployment uses GitHub Actions: a repository `CNAME` file does not configure
the domain; use the Pages settings. All site asset URLs are relative, so the
same files work at either the default project URL or a custom domain.

See [GitHub's custom domain documentation](https://docs.github.com/en/pages/configuring-a-custom-domain-for-your-github-pages-site/managing-a-custom-domain-for-your-github-pages-site).
