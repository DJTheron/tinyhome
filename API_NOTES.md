# What each thing returns

Two contracts you need: the LFM router, and the Homey Web API.

---

## 1. `model.route()` — LFM2.5-Encoder-350M-Prompt-Router

Not documented on the model card. Read from the cached `trust_remote_code` source:
`~/.cache/huggingface/hub/models--LiquidAI--LFM2.5-Encoder-350M-Prompt-Router/snapshots/*/modeling_lfm2_bidirectional.py:279`

```python
def route(self, text, routes, tokenizer, threshold=None)
```

**Returns:** a `list` of `dict`, **already sorted by score descending**:

```python
[{"route": "Daniel Desk Light", "score": 0.81},
 {"route": "Daniel Ceiling Light", "score": 0.12},
 {"route": "Daniel Bedside Light", "score": 0.07}]
```

So `results[0]` is the winner. The `max(data, key=...)` in `lfm-router.py:13` is redundant.

### The important bit: scores are a softmax

Source line 305: `probs = logits.softmax(dim=-1)`. Consequences:

- **Scores sum to 1.0 across your routes.** It is a *relative* ranking, not an absolute confidence.
- **A fixed threshold does not mean the same thing as route count changes.** With 3 routes,
  a garbage prompt floors around 0.33 each. With 20 routes it floors around 0.05, and a
  genuinely good match will also score lower purely from the bigger denominator.
- **There is no built-in "none of the above."** Softmax must distribute all the mass among
  the routes you supplied. Out-of-domain rejection by score alone is inherently weak.

### Two things that follow

**`threshold=` is already a parameter.** Pass it and routes below it are filtered out; if
everything is below, you get `[]`. That empty list *is* the rejection path from NOTES.md #1 —
you don't have to build it.

**Better: add an explicit sink route.** Something like `"Not a device command"` as a real
entry in the routes list. Since softmax has to put the mass somewhere, giving it a legitimate
home is far more reliable than thresholding. Then: if `results[0]["route"]` is the sink, bail.
Use both together.

### The prompt it builds internally

```
Categories:
- Daniel Ceiling Light
- Daniel Desk Light
- Daniel Bedside Light

Text:
turn the desk light on
```

Gotchas visible in the source:
- **Routes must not contain newlines.** `_category_ranges` (line 267) assumes one line per
  route and walks `pos = end + 1`. A newline silently misaligns every subsequent route's span.
- Leading/trailing whitespace in a route name becomes part of its token span. (Your
  "Living room TV " has a trailing space — not a bedroom device, but be aware.)
- `route()` is already decorated `@torch.no_grad()`. Don't wrap it again.

### Route naming

All three of your devices are now `Daniel <X> Light` — they share 2 of 3 words, and the whole
decision rests on Ceiling/Desk/Bedside. Consider routing on just the distinguishing part
(`"Ceiling Light"`, `"Desk Light"`, `"Bedside Light"`) and keeping a map back to the real
device name. One room means "Daniel" carries zero discriminating signal, it only dilutes.

---

## 2. Homey Web API

### `GET /api/manager/zones/zone`

Dict keyed by zone UUID. Each: `id`, `name`, `parent` (zone UUID, `null` for Home).

Yours: `Daniel’s Bedroom` = `4cdb0219-bc77-41e8-8fbd-79acd670f01f`

### `GET /api/manager/devices/device`

Dict keyed by device UUID. Fields that matter to you:

| Field | Type | Notes |
|---|---|---|
| `id` | string | |
| `name` | string | `"Daniel Desk Light"` |
| `zone` | string | zone UUID — filter on this, never on zone *name* |
| `class` | string | `"light"`, `"socket"`, `"speaker"`, `"sensor"`, `"other"` |
| `available` | boolean | **device is reachable right now** |
| `ready` | boolean | initialisation complete |
| `capabilities` | array of strings | `["onoff", "dim", "light_temperature"]` |
| `capabilitiesObj` | object keyed by capability id | see below |

Also present but not useful here: `driverId`, `driverUri`, `uri`, `icon`, `color`, `note`,
`settings`, `energy`, `insights`, `flags`, `images`, `ui`, `lastSeenAt`, `virtualClass`,
`unavailableMessage`, `warningMessage`.

### `capabilitiesObj[<capability_id>]`

| Field | Notes |
|---|---|
| `id` | e.g. `"onoff"` |
| `title` | human-readable |
| `type` | `"boolean"` / `"number"` / `"string"` / `"enum"` |
| `value` | **the current value** |
| `lastUpdated` | timestamp |
| `min` / `max` / `step` / `decimals` | numeric constraints |
| `units` | e.g. `"°C"`, `"W"` |
| `getable` / `setable` | whether you may read / write it |

**Two things this hands you for free:**

1. `capabilitiesObj["onoff"]["value"]` is the current on/off state. That's the toggle fallback
   from NOTES.md #3 — no extra endpoint needed, it's in the same payload.
2. `capabilities` tells you whether a device supports `light_temperature` *before* you try to
   set it. That's your guard for the colour-temp feature (NOTES.md #4).

Also check `available` when building routes — an offline light is still listed, and you don't
want to offer routes for dead devices.

### `PUT /api/manager/devices/device/{deviceId}/capability/{capabilityId}`

Body: `{"value": <value>}`, or `{"value": <value>, "duration": <ms>}` for a gradual transition
(nice for `dim` and `light_temperature` fades).

`capabilityId` is **any setable capability**, not just `onoff` — the same endpoint sets `dim`
and `light_temperature`. So `onoffcontrol()` generalises by taking the capability id as an
argument instead of hardcoding `onoff`.

### Your bedroom devices

| Device ID | name | capabilities |
|---|---|---|
| `14d0622d-deb2-447a-84cb-144677d0ee26` | Daniel Ceiling Light | onoff, dim, light_temperature, light_mode, light_hue, light_saturation |
| `249c9786-9390-4ffd-a4ee-4b908dbaed3a` | Daniel Desk Light | onoff, dim, light_temperature |
| `c0a1f8b6-0218-4b86-81c3-077997037b5d` | Daniel Bedside Light | onoff, dim, light_temperature, light_hue, light_saturation, light_mode |

`light_temperature` is a 0–1 float in your live data (0.34 / 0.67 / 0.75 / 1). `light_mode` is
`"temperature"` or `"color"`. Athom don't document whether setting `light_temperature` also
requires setting `light_mode` — test it on the Desk Light, which has no `light_mode` at all,
versus the Bedside Light, which does.

---

## 3. Shape of the code

**Startup, once:**
1. GET zones, GET devices
2. keep devices where `zone == BEDROOM_ZONE_ID` and `available` and `"onoff" in capabilities`
3. build `routes` = those names (+ the sink route)
4. keep `route string -> device id` and `route string -> capabilities` maps

**Per command:**
1. transcript out of Whisper
2. `results = model.route(transcript, routes, tokenizer, threshold=T)`
3. bail if `not results`, or if `results[0]["route"]` is the sink
4. look up the device id from the winning route
5. state: word-boundary keyword match on the transcript. If no state word → re-GET that one
   device and invert `capabilitiesObj["onoff"]["value"]`
6. PUT the capability

**While tuning:** log the *entire* results list, not just the top one. The gap between top-1
and top-2 is the number that tells you whether a threshold is viable — a 0.81/0.12 split is
healthy, a 0.38/0.34 split means the router is guessing regardless of what the top score is.
