import requests


class PostEndpoint:
    def __init__(self, url: str, session: requests.Session | None = None):
        self.url = url
        self.session = session or requests.Session()

    def post(self, json=None, data=None, headers=None) -> requests.Response:
        return self.session.post(
            self.url,
            json=json,
            data=data,
            headers=headers,
        )


class GetEndpoint:
    def __init__(self, url: str, session: requests.Session | None = None):
        self.url = url
        self.session = session or requests.Session()

    def get(self) -> requests.Response:
        return self.session.get(self.url)
