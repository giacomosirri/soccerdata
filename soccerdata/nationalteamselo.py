"""Scraper for eloratings.net"""

from datetime import timedelta, date
from pathlib import Path
from typing import Callable, Optional, Union
import numpy as np
import pandas as pd
import csv
from ._common import BaseRequestsReader, standardize_colnames
from ._config import DATA_DIR, NOCACHE, NOSTORE

NT_ELO_DATADIR = DATA_DIR / "NationalTeamsElo"
NT_ELO_API = "https://eloratings.net"

def string_to_num(item) -> float:
    if type(item) == str:
        item = item.replace("−", "-")
    return float(item) if item != "-" else np.nan

class NationalTeamsElo(BaseRequestsReader):
    """Provides pd.DataFrames from scraping endpoints at https://eloratings.net.

    Data will be downloaded as necessary and cached locally in
    ``~/soccerdata/data/NationalTeamsElo``.
    
    Parameters
    ----------
    proxy : 'tor' or or dict or list(dict) or callable, optional
        Use a proxy to hide your IP address. Valid options are:
            - "tor": Uses the Tor network. Tor should be running in
              the background on port 9050.
            - str: The address of the proxy server to use.
            - list(str): A list of proxies to choose from. A different proxy will
              be selected from this list after failed requests, allowing rotating
              proxies.
            - callable: A function that returns a valid proxy. This function will
              be called after failed requests, allowing rotating proxies.
    no_cache : bool
        If True, will not use cached data.
    no_store : bool
        If True, will not store downloaded data.
    data_dir : Path
        Path to directory where data will be cached.
    """
    def __init__(
        self,
        proxy: Optional[Union[str, list[str], Callable[[], str]]] = None,
        no_cache: bool = NOCACHE,
        no_store: bool = NOSTORE,
        data_dir: Path = NT_ELO_DATADIR,
    ):
        """Initialize a new eloratings.net reader."""
        super().__init__(proxy=proxy, no_cache=no_cache, no_store=no_store, data_dir=data_dir)
    
    def _read_country_codes(self) -> dict[str, list[str]]:
        filepath = self.data_dir / f"team_codes.csv"
        url = f"{NT_ELO_API}/en.teams.tsv"

        self.get(url, filepath)

        with open(filepath, mode="r", encoding="utf-8") as csvfile:
            list_of_dicts = list(csv.DictReader(csvfile, dialect="excel-tab", fieldnames=["Code"], 
                                                restkey="Country"))
        dict = {item["Code"]: item["Country"] if isinstance(item["Country"], list) else [item["Country"]]
                for item in list_of_dicts}
        return dict
    
    def _read_competition_codes(self) -> dict[str, list[str]]:
        filepath = self.data_dir / f"competition_codes.csv"
        url = f"{NT_ELO_API}/en.tournaments.tsv"

        self.get(url, filepath)

        with open(filepath, mode="r", encoding="utf-8") as csvfile:
            list_of_dicts = list(csv.DictReader(csvfile, dialect="excel-tab", fieldnames=["Code"], 
                                                restkey="Competition"))
        dict = {item["Code"]: item["Competition"] if isinstance(item["Competition"], list) else [item["Competition"]]
                for item in list_of_dicts}
        return dict
    
    def _read_ranking(self, url, filepath) -> pd.DataFrame:
        cols = ["Rank", "Country", "Elo", "HighestRank", "HighestElo", "LowestRank", "LowestElo",
                "AverageRank", "AverageElo", "1YearDeltaPos", "1YearDeltaElo", "TotalMatches",
                "HomeMatches", "AwayMatches", "NeutralFieldMatches", "WonMatches", "LostMatches",
                "DrawnMatches", "GoalsFor", "GoalsAgainst"]

        data = self.get(url, filepath)
        
        df = pd.read_csv(data, sep=r'\t', names=cols, engine="python",
                         usecols=[0,2,3,4,5,6,7,8,9,14,15,22,23,24,25,26,27,28,29,30],
                         keep_default_na=False)
        
        codes_dict = self._read_country_codes()
        df["Country"] = df["Country"].apply(lambda country: codes_dict[country][0])

        return (
            df.apply(lambda series: series.apply(string_to_num) if series.name != "Country" else series)
              .pipe(standardize_colnames)
        )

    def read_current_ranking(self, conf: str | None) -> pd.DataFrame:
        """Retrieve current ELO scores for all national teams.

        Elo scores are calculated throughout football history, with earliest entries 
        dating back to 1872 (!).

        Parameters
        ----------
        conf : str
            One of the six FIFA-affiliated confederations
            (AFC, CAF, CONCACAF, CONMEBOL, OFC, UEFA) or 'Unaffiliated'.
            If 'None' is provided, then the returned DataFrame will list
            every national team in the world.
        max_age : int for age in days, or timedelta object
            The max. age of locally cached file before re-download.

        Returns
        -------
        pd.DataFrame
        """
        if conf is None:
            filepath = self.data_dir / "current_ranking.csv"
            url = f"{NT_ELO_API}/World.tsv"
        else:
            filepath = self.data_dir / f"{conf}.csv"
            url = f"{NT_ELO_API}/{conf}.tsv"
        return self._read_ranking(url, filepath)
    
    def read_end_of_year_ranking(self, year: str|int):
        """Retrieve ELO scores for all national teams on 31 December of the given year.

        Returns
        -------
        pd.DataFrame
        """
        year = int(year)
        if year < 1901 or year >= date.today().year:
            raise ValueError("There is no end of year ranking available for the given year.")
        filepath = self.data_dir / f"{year}_ranking.csv"
        url = f"{NT_ELO_API}/{year}.tsv"

        return self._read_ranking(url, filepath)
        
    def read_country_history(self, country: str, max_age: Union[int, timedelta] = 1) -> pd.DataFrame:
        """Retrieve full ELO history for one country's national team.

        For the exact spelling of a country's name, check the result of
        :func:`~soccerdata.NationalTeamsElo.read_current_ranking` or `eloratings.net
        <http://eloratings.net/>`__.

        Parameters
        ----------
        country : str
            The country's name.
        max_age : int for age in days, or timedelta object
            The max. age of locally cached file before re-download.

        Raises
        ------
        TypeError
            If max_age is not an integer or timedelta object.
        ValueError
            If no ratings for the given team are available.

        Returns
        -------
        pd.DataFrame
        """
        # Since URLs cannot contain spaces, multi-word country names need additional underscores.
        country_url = country.replace(" ", "_")
        filepath = self.data_dir / f"{country_url}.csv"
        url = f"{NT_ELO_API}/{country_url}.tsv"

        cols = ["Year", "Month", "Day", "HomeTeam", "AwayTeam", "HomeTeamGoals", "AwayTeamGoals",
                "Competition", "PlayedIn", "HomeEloDelta", "HomeEloPostMatch", "AwayEloPostMatch",
                "HomeRankDelta", "AwayRankDelta", "HomeRankPostMatch", "AwayRankPostMatch"]

        try:
            data = self.get(url, filepath, max_age)
        except ConnectionError:
            raise ValueError("No ratings are available for the given country.")
        
        df = pd.read_csv(data, sep=r'\t', names=cols, engine="python", keep_default_na=False)

        # Create date column by merging year, month and day.
        df.insert(0, "MatchDate", pd.to_datetime(arg=df.loc[:, ["Year", "Month", "Day"]],
                                                 errors="coerce"))
        df = df.drop(["Year", "Month", "Day"], axis=1)

        df["PlayedIn"] = df["PlayedIn"].replace("", np.nan).combine_first(other=df["HomeTeam"])
        country_codes = self._read_country_codes()
        df[["HomeTeam", "AwayTeam", "PlayedIn"]] = df[["HomeTeam", "AwayTeam", "PlayedIn"]].apply(
            lambda series: series.apply(lambda code: country_codes[code][0]))

        competition_codes = self._read_competition_codes()
        df[["Competition"]] = df[["Competition"]].apply(lambda series: series.apply(
            lambda comp: competition_codes[comp][0]))

        non_number_cols = ["MatchDate", "HomeTeam", "AwayTeam", "Competition", "PlayedIn"]
        return (
            df.apply(lambda series: series.apply(string_to_num) if series.name not in non_number_cols else series)
              .pipe(standardize_colnames)
        )
