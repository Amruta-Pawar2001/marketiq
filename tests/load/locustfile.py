from locust import HttpUser, between, task


class MarketIQUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def predict_discount(self):
        self.client.post(
            "/predict_discount",
            json={
                "category": "electronics",
                "actual_price": 2499,
                "discounted_price": 1699,
                "rating": 4.2,
                "rating_count": 1240,
            },
        )

    @task
    def answer_question(self):
        self.client.post(
            "/answer_question",
            json={"question": "What USB cables are available under Rs500?", "top_k": 5},
        )
