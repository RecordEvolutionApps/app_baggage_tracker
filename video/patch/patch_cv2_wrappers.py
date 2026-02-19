"""
Patch cv2 Python wrappers after a source build without the gapi module.

When gapi is absent or partially compiled two files crash on `import cv2`:
  1. cv2/gapi/__init__.py – unconditionally assigns cv.gapi_wip_gst_GStreamerPipeline
  2. cv2/typing/__init__.py – references cv2.GMat / GOpaqueT / GArrayT at module level

Both are guarded here so the import succeeds regardless of whether gapi was
compiled.  When gapi IS compiled the guards are harmless no-ops.
"""
import glob
import re
import sys

# ── patch cv2/gapi/__init__.py ───────────────────────────────────────────────
for path in glob.glob('/usr/local/lib/python3*/site-packages/cv2/gapi/__init__.py'):
    txt = open(path).read()
    old = 'cv.gapi.wip.GStreamerPipeline = cv.gapi_wip_gst_GStreamerPipeline'
    new = ('if hasattr(cv, "gapi_wip_gst_GStreamerPipeline"): '
           'cv.gapi.wip.GStreamerPipeline = cv.gapi_wip_gst_GStreamerPipeline')
    if old in txt:
        open(path, 'w').write(txt.replace(old, new))
        print(f'Patched gapi/__init__.py: {path}')
    else:
        print(f'gapi/__init__.py already patched or pattern not found: {path}')

# ── patch cv2/typing/__init__.py ─────────────────────────────────────────────
for path in glob.glob('/usr/local/lib/python3*/site-packages/cv2/typing/__init__.py'):
    content = open(path).read()

    # Guard the bare `import cv2.gapi.wip.draw`
    content = content.replace(
        'import cv2.gapi.wip.draw',
        'try:\n    import cv2.gapi.wip.draw\nexcept Exception:\n    pass',
    )

    # Guard top-level type-alias assignments referencing missing gapi symbols
    for alias in ('GProtoArg', 'GRunArg', 'GMetaArg', 'Prim', 'GTypeInfo',
                  'ExtractArgsCallback', 'ExtractMetaCallback'):
        pattern = re.compile(
            r'^(' + re.escape(alias) + r'\s*=\s*.+)$', re.MULTILINE
        )
        def make_guarded(m, alias=alias):
            return (f'try:\n    {m.group(1)}\n'
                    f'except AttributeError:\n    {alias} = _typing.Any  # gapi not compiled')
        content, n = pattern.subn(make_guarded, content)
        if n:
            print(f'  Guarded {alias} in {path}')

    open(path, 'w').write(content)

# ── verify ────────────────────────────────────────────────────────────────────
for mod in list(sys.modules.keys()):
    if mod.startswith('cv2'):
        del sys.modules[mod]

import cv2  # noqa: E402
print(f'cv2 import OK: {cv2.__version__}')
gst = [l for l in cv2.getBuildInformation().split('\n') if 'GStreamer' in l]
assert any('YES' in l for l in gst), 'GStreamer not enabled in cv2'
print(f'GStreamer: {gst}')
