import time
import urllib


class URLLoader:
    """Retrieves a single page and report the URL and contents" """

    def __init__(
        self, url: str = "http://www.foxnews.com/", sleep: int = 3
    ) -> None:
        self.url = url
        self.sleep = sleep

    def run(self):
        time.sleep(self.sleep)
        with urllib.request.urlopen(url=self.url) as conn:
            return conn.read()
