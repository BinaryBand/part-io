# partio GUI app (Flutter)

Talks to the local `gui/server` backend over HTTP/WebSocket -- see `../server/README.md` for how to run that first.

## System prerequisites (Linux desktop)

Flutter's own desktop build tooling:

```sh
sudo apt-get install -y clang cmake ninja-build pkg-config libgtk-3-dev
```

`media_kit` (audio playback) additionally requires the system's `libmpv` on Linux -- unlike other platforms, media_kit does not bundle it automatically here:

```sh
sudo apt-get install -y libmpv-dev mpv
```

Confirmed on this machine (2026-09-16): `media_kit_libs_linux`'s CMakeLists only vendors the `mimalloc` allocator override, nothing mpv-related -- `libmpv` genuinely has to come from the system. Without it, `MediaKit.ensureInitialized()` throws `Cannot find libmpv at the usual places` at runtime (confirmed via both `flutter test` and a real `flutter run -d linux`).

## Run

```sh
fvm flutter run -d linux
```

## Gate

```sh
fvm flutter analyze && fvm flutter test
```

Widget/unit tests intentionally do not depend on real native audio playback (no `libmpv` requirement) -- from Phase 1 onward, `audio_player_service.dart` sits behind an interface so tests inject a fake. Real playback is verified by actually running the app, not by the automated test gate.
