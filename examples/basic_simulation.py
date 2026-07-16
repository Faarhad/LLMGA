"""Minimal example of using the discrete-event simulation engine."""

from desim import Event, Simulation


def main() -> None:
    sim = Simulation()

    def on_any(state, event):
        log = state.get("log", [])
        log.append(f"t={event.time:.1f}: {event.name}")
        state.set("log", log)

    sim.schedule(Event(time=5.0, name="late", handler=on_any))
    sim.schedule(Event(time=1.0, name="early", handler=on_any))
    sim.schedule(Event(time=3.0, name="middle", handler=on_any))

    processed = sim.run()
    print(f"processed={processed}")
    for line in sim.state.get("log", []):
        print(line)


if __name__ == "__main__":
    main()
