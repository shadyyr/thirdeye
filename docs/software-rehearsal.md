# Software rehearsal toolkit

Third Eye can rehearse its decision logic without a Jetson, cameras, ESP32, or belt. Named deterministic scenarios supply the same normalized mock detections on every run, making tuning and regression testing meaningful.

## Scenarios

```bash
thirdeye --list-scenarios
```

| Scenario | Expected behavior |
| --- | --- |
| `front_right_approach` | An approaching front-right person triggers `RIGHT`. |
| `front_stationary` | A stationary person stays quiet. |
| `front_moving_away` | A retreating person stays quiet. |
| `rear_left_approach` | A rear-left approach triggers the configured physical side. |
| `competing_people` | An approaching person beats a more visually prominent stationary person. |
| `approach_then_disappear` | A valid alert is followed by a safe `STOP` when the person leaves. |

Run any scenario:

```bash
thirdeye --scenario competing_people --frames 0
```

`--frames 0` allows the finite scenario to run to its natural end. The default scenario is `front_right_approach`.

## Dashboard rehearsal

After installing the optional dashboard adapter, use a half-second replay cadence so the state is readable while watching the terminal or local browser:

```bash
pip install -e ".[dashboard]"
thirdeye --scenario rear_left_approach --frames 0 --dashboard --replay
```

Open `http://127.0.0.1:8000`. The dashboard displays scenario name, selected event, haptic output, the pipeline snapshot, and timing. It remains a decision-visibility tool rather than a final video-overlay frontend.
After a finite mock scenario completes, the dashboard stays open with its final state until `Ctrl+C`.

## Orientation calibration

Before a belt is connected, inspect the configuration-implied mapping:

```bash
thirdeye --calibrate-orientation
```

When hardware is ready, put a person at the listed left/center/right positions for each camera and compare the physical motor that fires. If the rear view is reversed, create `config/local.json` with the appropriate `camera_mappings` or `mirror_horizontal` override. Do not change `config/default.json` for one physical rig.

## Decision acceptance tests

The end-to-end tests run the actual mock camera → detector → tracker → motion → salience → haptic controller path. Run them before modifying scoring thresholds or direction configuration:

```bash
python -m unittest discover -s tests -v
```

They protect the demo-critical expectations that stationary/retreating people are quiet, approaching people alert in the right zone, the strongest event wins, disappearing targets stop safely, and rear orientation remains configurable.
