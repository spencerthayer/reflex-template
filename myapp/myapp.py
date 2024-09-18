import reflex as rx
import reflex.recharts as recharts  # Import recharts module
from myapp.analysis import main
import asyncio
from typing import Dict, Any, List

class PollState(rx.State):
    polling_data: Dict[str, Any] = {}
    is_loading: bool = True

    async def periodic_update(self):
        while True:
            self.is_loading = True
            try:
                data = await main()
                if isinstance(data, dict):
                    self.polling_data = data
                else:
                    print(f"Unexpected data format from main(): {type(data)}")
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

    @rx.var
    def periods(self) -> List[str]:
        return list(self.polling_data.keys())

def poll_chart(period: str, result: Dict[str, Any]) -> rx.Component:
    chart_data = [
        {
            "candidate": candidate,
            "score": score[0],
            "margin": score[1]
        }
        for candidate, score in result.get('polling_metrics', {}).items()
    ]

    return rx.vstack(
        rx.heading(f"Polling Results for {period}", size="md"),
        recharts.composed_chart(
            recharts.bar(data_key="score", fill="#8884d8"),
            recharts.error_bar(data_key="margin", width=4, stroke="red"),
            recharts.x_axis(data_key="candidate"),
            recharts.y_axis(),
            recharts.cartesian_grid(stroke_dasharray="3 3"),
            recharts.graphing_tooltip(),
            data=chart_data,
            height=300,
            width="100%",
        ),
        rx.text(f"OOB Variance: {result.get('oob_variance', 0):.2f}"),
        margin_bottom="1em"
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
                        PollState.periods,
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
