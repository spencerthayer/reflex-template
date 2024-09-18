import reflex as rx
from myapp.analysis import main
import asyncio
from typing import Dict, Any

class PollState(rx.State):
    polling_data: Dict[str, Any] = {}
    is_loading: bool = True

    async def periodic_update(self):
        while True:
            self.is_loading = True
            try:
                data = main()
                if isinstance(data, dict):
                    self.polling_data = data
                else:
                    self.polling_data = {}
            except Exception as e:
                print(f"Error updating poll data: {e}")
                self.polling_data = {}
            finally:
                self.is_loading = False
            await asyncio.sleep(600)

    @rx.background
    async def start_periodic_update(self):
        await self.periodic_update()

def poll_chart(period: str, result: Dict[str, Any]) -> rx.Component:
    return rx.box(
        rx.text(f"{period}:"),
        rx.chart(
            data=[{"candidate": candidate, "score": score[0], "margin": score[1]} 
                  for candidate, score in result.get('polling_metrics', {}).items()],
            type="bar",
            x="candidate",
            y="score",
            title=f"Polling Results for {period}",
        ),
        rx.text(f"OOB Variance: {result.get('oob_variance', 0):.2f}")
    )

def index() -> rx.Component:
    return rx.vstack(
        rx.heading("Election Polling Analysis"),
        rx.cond(
            PollState.is_loading,
            rx.text("Loading data..."),
            rx.cond(
                PollState.polling_data != {},
                rx.vstack(
                    rx.foreach(
                        PollState.polling_data.keys(),  # Iterate over keys
                        lambda period: poll_chart(str(period), PollState.polling_data[period])
                    )
                ),
                rx.text("No data available.")
            )
        ),
        on_mount=PollState.start_periodic_update
    )

app = rx.App(state=PollState)
app.add_page(index)