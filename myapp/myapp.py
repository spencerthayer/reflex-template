import reflex as rx
from analysis import main  # Import the refactored polling function

class PollState(rx.State):
    polling_data = {}  # Store polling data

    @rx.timer(interval=600)  # Fetch data every 10 minutes
    def update_data(self):
        self.polling_data = main()  # Call the main function to get data

def index() -> rx.Component:
    return rx.vstack(
        rx.heading("Election Polling Analysis"),
        rx.foreach(
            PollState.polling_data.items(),
            lambda period, result: rx.box(
                rx.text(f"{period}:"),
                rx.chart(
                    data=[{"candidate": candidate, "score": score[0], "margin": score[1]} for candidate, score in result['polling_metrics'].items()],
                    chart_type="bar",
                    x="candidate",
                    y="score",
                    title=f"Polling Results for {period}",
                ),
                rx.text(f"OOB Variance: {result['oob_variance']:.2f}")
            )
        )
    )

app = rx.App()
app.add_page(index)
app.compile()