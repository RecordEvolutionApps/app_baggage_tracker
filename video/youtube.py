"""YouTube URL resolution via yt-dlp binary or Python library."""
from __future__ import annotations

import logging
import os

logger = logging.getLogger('youtube')


def get_youtube_video(url, height=None):
    """Resolve a YouTube URL into a direct playable stream URL.

    Uses the standalone yt-dlp binary (/usr/local/bin/yt-dlp) which bundles its
    own Python and stays up-to-date with YouTube's nsig extraction changes,
    independent of the system Python 3.10 constraint.

    Falls back to the yt_dlp Python library if the binary is not available.
    """
    import shutil
    import subprocess
    import json as _json

    cookie_file = '/data/cookies.txt'

    # Prefer the highest-resolution video-only stream.  YouTube serves
    # high-quality formats (720p+) only as separate video/audio DASH tracks;
    # the muxed "best" format is often just 360p.  OpenCV only needs the
    # video track, so video-only is fine.
    #
    # Exclude AV1 (vcodec!=av01): Jetson Xavier/Orin have no AV1 HW decoder,
    # so FFmpeg falls back to libaom software decode which is too slow for
    # real-time and produces corrupt-frame errors.  Prefer H.264/H.265.
    if height and height > 0:
        format_str = (f'bestvideo[height<={height}][vcodec!=av01][ext=mp4]'
                      f'/bestvideo[height<={height}][vcodec!=av01]'
                      f'/best[height<={height}][vcodec!=av01]'
                      f'/bestvideo[vcodec!=av01][ext=mp4]'
                      f'/bestvideo[vcodec!=av01]/best')
    else:
        format_str = ('bestvideo[vcodec!=av01][ext=mp4]'
                      '/bestvideo[vcodec!=av01]/best')

    logger.info('[yt-dlp] Resolving: %s (format=%s)', url, format_str)

    yt_dlp_bin = shutil.which('yt-dlp')
    if yt_dlp_bin is None:
        logger.warning('[yt-dlp] Standalone binary not found — falling back to Python yt_dlp library')
        return _get_youtube_video_lib(url, height)

    # Build the command:
    #   -f <format>       : format selector
    #   -g                : print the resolved stream URL to stdout
    #   --print %(width)sx%(height)s : print WIDTHxHEIGHT on a second line
    #   --no-warnings     : keep stderr clean
    #   --no-playlist     : single video only
    cmd = [
        yt_dlp_bin,
        '-f', format_str,
        '--print', '%(url)s',                   # line 1: stream URL
        '--print', '%(width)sx%(height)s',      # line 2: WIDTHxHEIGHT
        '--no-warnings',
        '--no-playlist',
    ]

    if os.path.isfile(cookie_file):
        logger.info('[yt-dlp] Using cookie file: %s', cookie_file)
        cmd += ['--cookies', cookie_file]

    cmd.append(url)

    logger.info('[yt-dlp] Running: %s', ' '.join(cmd))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        logger.error('[yt-dlp] Timed out after 30 s')
        return {'url': url, 'width': 0, 'height': 0}

    if result.returncode != 0:
        logger.error('[yt-dlp] Failed (exit %d): %s', result.returncode, result.stderr.strip())
        # Fall back to the Python library in case the binary is outdated too
        return _get_youtube_video_lib(url, height)

    lines = result.stdout.strip().splitlines()
    stream_url = lines[0] if len(lines) >= 1 else ''
    resolution_str = lines[1] if len(lines) >= 2 else '0x0'

    width, height_val = 0, 0
    try:
        parts = resolution_str.split('x')
        width = int(parts[0]) if parts[0] not in ('NA', 'None', '') else 0
        height_val = int(parts[1]) if len(parts) > 1 and parts[1] not in ('NA', 'None', '') else 0
    except (ValueError, IndexError):
        pass

    if stream_url:
        logger.info('[yt-dlp] Resolved %dx%d — URL starts with: %s',
                     width, height_val, stream_url[:120])
    else:
        logger.error('[yt-dlp] No URL in stdout: %s', result.stdout[:200])

    return {
        'url': stream_url or url,
        'width': width,
        'height': height_val,
    }


def _get_youtube_video_lib(url, height=None):
    """Fallback: use the yt_dlp Python library (may have stale nsig extraction)."""
    import yt_dlp

    # Prefer video-only DASH streams for higher resolution (see get_youtube_video).
    # Exclude AV1 — no HW decoder on Jetson; software libaom is too slow.
    if height and height > 0:
        format_str = (f'bestvideo[height<={height}][vcodec!=av01][ext=mp4]'
                      f'/bestvideo[height<={height}][vcodec!=av01]'
                      f'/best[height<={height}][vcodec!=av01]'
                      f'/bestvideo[vcodec!=av01][ext=mp4]'
                      f'/bestvideo[vcodec!=av01]/best')
    else:
        format_str = ('bestvideo[vcodec!=av01][ext=mp4]'
                      '/bestvideo[vcodec!=av01]/best')

    cookie_file = '/data/cookies.txt'

    ydl_opts = {
        'format': format_str,
        'quiet': True,
        'no_warnings': True,
        'extractor_args': {
            'youtube': {
                'player_client': ['mweb', 'android'],
            },
        },
    }

    if os.path.isfile(cookie_file):
        ydl_opts['cookiefile'] = cookie_file

    logger.info('[yt_dlp-lib] Resolving: %s (format=%s)', url, format_str)

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

        stream_url = info.get('url', '')

        if not stream_url:
            for fmt in info.get('requested_formats', []):
                if fmt.get('vcodec', 'none') != 'none' and fmt.get('url'):
                    stream_url = fmt['url']
                    break

        if not stream_url:
            for fmt in reversed(info.get('formats', [])):
                if (fmt.get('vcodec', 'none') != 'none' and
                    fmt.get('acodec', 'none') != 'none' and
                    fmt.get('url')):
                    stream_url = fmt['url']
                    if fmt.get('width'):
                        info['width'] = fmt['width']
                    if fmt.get('height'):
                        info['height'] = fmt['height']
                    break

        if stream_url:
            logger.info('[yt_dlp-lib] Resolved — URL len=%d', len(stream_url))
        else:
            logger.error('[yt_dlp-lib] Could not find any playable URL')

        return {
            'url': stream_url or url,
            'width': info.get('width', 0),
            'height': info.get('height', 0),
        }
