import json
import logging

from locust import HttpUser, task

# Reduce logging to improve performance
logging.getLogger("urllib3").setLevel(logging.WARNING)


class BGEApi(HttpUser):
    @task
    def info(self):
        """Test health endpoint"""
        with self.client.get("/", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Model info failed: {response.status_code}")

    @task
    def dense(self):
        """Test single prediction endpoint"""
        with self.client.post(
                "/bge-m3/dense", json={"texts": ["string"]}, catch_response=True, timeout=10
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "dense_vecs" in data:
                        response.success()
                    else:
                        response.failure("Invalid response format")
                except json.JSONDecodeError:
                    response.failure("Failed to parse JSON")
            elif response.status_code == 503:
                response.failure("Service unavailable")
            else:
                response.failure(f"Status code: {response.status_code}")

    @task
    def lexical(self):
        """Test single prediction endpoint"""
        with self.client.post(
                "/bge-m3/lexical", json={"texts": ["string"]}, catch_response=True, timeout=10
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "lexical_weights" in data:
                        response.success()
                    else:
                        response.failure("Invalid response format")
                except json.JSONDecodeError:
                    response.failure("Failed to parse JSON")
            elif response.status_code == 503:
                response.failure("Service unavailable")
            else:
                response.failure(f"Status code: {response.status_code}")
    @task
    def colbert(self):
        """Test single prediction endpoint"""
        with self.client.post(
                "/bge-m3/colbert", json={"texts": ["string"]}, catch_response=True, timeout=10
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "colbert_vecs" in data:
                        response.success()
                    else:
                        response.failure("Invalid response format")
                except json.JSONDecodeError:
                    response.failure("Failed to parse JSON")
            elif response.status_code == 503:
                response.failure("Service unavailable")
            else:
                response.failure(f"Status code: {response.status_code}")

    @task
    def all(self):
        """Test single prediction endpoint"""
        with self.client.post(
                "/bge-m3/all", json={"texts": ["string"]}, catch_response=True, timeout=10
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "colbert_vecs" in data:
                        response.success()
                    else:
                        response.failure("Invalid response format")
                except json.JSONDecodeError:
                    response.failure("Failed to parse JSON")
            elif response.status_code == 503:
                response.failure("Service unavailable")
            else:
                response.failure(f"Status code: {response.status_code}")
