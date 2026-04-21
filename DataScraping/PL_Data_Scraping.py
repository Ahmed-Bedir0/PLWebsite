import io
import time
import random
import logging
import requests
import pandas as pd
from bs4 import BeautifulSoup, Comment

SEASON_URL = "https://fbref.com/en/comps/9/2024-2025/2024-2025-Premier-League-Stats"
REQUEST_DELAY_MIN = 4
REQUEST_DELAY_MAX = 7
OUTPUT_FILE = "prem_stats.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-GB,en;q=0.9",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(HEADERS)
    return session


def get_with_retry(session: requests.Session, url: str, retries: int = 3):
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=30)
            if resp.status_code == 200:
                return resp
            if resp.status_code == 429:
                log.warning("Rate limited (429). Cooling down for 60s...")
                time.sleep(60)
                continue
            log.warning("Attempt %d: HTTP %d → %s", attempt + 1, resp.status_code, url)
        except requests.RequestException as e:
            log.warning("Attempt %d: Connection error → %s", attempt + 1, str(e))
        time.sleep(5 * (attempt + 1))
    return None


def extract_tables_from_comments(soup: BeautifulSoup) -> BeautifulSoup:
    comments = soup.find_all(string=lambda text: isinstance(text, Comment))
    for c in comments:
        if "table" in c:
            return BeautifulSoup(c, "lxml")
    return soup


def fetch_team_urls(session: requests.Session) -> list[str]:
    log.info("Fetching team URLs from %s", SEASON_URL)
    resp = get_with_retry(session, SEASON_URL)
    if not resp:
        raise RuntimeError("Could not reach standings page.")

    soup = BeautifulSoup(resp.text, "lxml")
    table = soup.find("table", class_="stats_table")
    if table is None:
        raise RuntimeError("Standings table not found.")

    hrefs = [a.get("href", "") for a in table.find_all("a")]
    squad_paths = list(dict.fromkeys(h for h in hrefs if "/squads/" in h))
    if not squad_paths:
        raise RuntimeError("No squad links found.")

    urls = [f"https://fbref.com{path}" for path in squad_paths]
    log.info("Found %d teams.", len(urls))
    return urls


def scrape_team(session: requests.Session, url: str) -> pd.DataFrame | None:
    team_name = url.split("/")[-1].replace("-Stats", "").replace("-", " ")
    resp = get_with_retry(session, url)
    if not resp:
        log.warning("Skipping %s — request failed", team_name)
        return None

    soup = BeautifulSoup(resp.text, "lxml")
    table = soup.find("table", id="stats_standard")

    if table is None:
        soup = extract_tables_from_comments(soup)
        table = soup.find("table", id="stats_standard")

    if table is None:
        log.warning("Skipping %s — stats table not found", team_name)
        return None

    df = pd.read_html(io.StringIO(str(table)))[0]

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(0)

    if "Player" in df.columns:
        df = df[df["Player"] != "Player"]
        df = df.dropna(subset=["Player"])

    df = df.reset_index(drop=True)
    df["Team"] = team_name

    log.info("  %-30s %d players collected", team_name, len(df))
    return df


def main():
    session = make_session()

    try:
        team_urls = fetch_team_urls(session)
    except RuntimeError as e:
        log.error(e)
        return

    all_teams = []

    for i, url in enumerate(team_urls, 1):
        log.info("[%d/%d] Scraping %s", i, len(team_urls), url.split("/")[-1])
        time.sleep(random.uniform(REQUEST_DELAY_MIN, REQUEST_DELAY_MAX))

        df = scrape_team(session, url)
        if df is not None:
            all_teams.append(df)

    if not all_teams:
        log.error("No data collected.")
        return

    stats = pd.concat(all_teams, ignore_index=True)
    stats = stats.drop_duplicates().reset_index(drop=True)

    stats.to_csv(OUTPUT_FILE, index=False)
    log.info("SUCCESS: Saved %d player rows → %s", len(stats), OUTPUT_FILE)


if __name__ == "__main__":
    main()
