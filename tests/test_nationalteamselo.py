import soccerdata as sd
import pandas as pd
import pytest

class TestRanking:
    """Tests for NationalTeamsElo.read_current_ranking"""

    def _check_dataframe(self, df: pd.DataFrame) -> None:
        assert(isinstance(df, pd.DataFrame))
        assert not df.empty
        assert set(df.columns) == set(["rank","country","elo","highest_rank","highest_elo","lowest_rank",
                                   "lowest_elo","average_rank","average_elo","1_year_delta_pos",
                                   "1_year_delta_elo","total_matches","home_matches","away_matches",
                                   "neutral_field_matches","won_matches","lost_matches","drawn_matches",
                                   "goals_for","goals_against"])
        # Can contain negative values.
        assert pd.api.types.is_numeric_dtype(df["1_year_delta_pos"])
        assert pd.api.types.is_numeric_dtype(df["1_year_delta_elo"])
        assert pd.api.types.is_string_dtype(df["country"])
        assert df["country"].nunique() == df["country"].size

    def test_current_ranking(self, nt: sd.NationalTeamsElo) -> None:
        df = nt.read_current_ranking(conf=None)
        self._check_dataframe(df)
        df = nt.read_current_ranking(conf="CONCACAF")

    def test_end_of_year_ranking(self, nt: sd.NationalTeamsElo) -> None:
        df = nt.read_end_of_year_ranking(2021)
        self._check_dataframe(df)
        df = nt.read_end_of_year_ranking("1955")
        self._check_dataframe(df)
        with pytest.raises(ValueError, match="There is no end of year ranking available for the given year."):
            _ = nt.read_end_of_year_ranking(2026)


class TestCountryHistory:
    """Tests for NationalTeamsElo.read_country_history"""

    def _check_dataframe(self, df: pd.DataFrame) -> None:
        assert(isinstance(df, pd.DataFrame))
        assert not df.empty
        assert set(df.columns) == set(["match_date", "home_team", "away_team",
                                       "home_team_goals", "away_team_goals", "competition",
                                       "played_in", "home_elo_delta", "home_elo_post_match",
                                       "away_elo_post_match", "home_rank_delta", "away_rank_delta",
                                       "home_rank_post_match", "away_rank_post_match"])
        assert pd.api.types.is_datetime64_dtype(df["match_date"])
        assert pd.api.types.is_numeric_dtype(df["home_elo_delta"])
        assert pd.api.types.is_numeric_dtype(df["home_rank_delta"])
        assert pd.api.types.is_numeric_dtype(df["away_rank_delta"])

    def test_existing_country(self, nt: sd.NationalTeamsElo) -> None:
        """Should return a valid DataFrame with ELO history for the given country."""
        df = nt.read_country_history("France")
        self._check_dataframe(df)

    def test_country_with_noncomplete_matchdate(self, nt: sd.NationalTeamsElo) -> None:
        """Should return a DataFrame with missing date for matches with incomplete date."""
        df = nt.read_country_history("Dominica")
        # The date of the first match is not fully stated (it appears as simply '1932').
        # Therefore, the year should be ignored and NaT should be used instead.
        assert df["match_date"][0] is pd.NaT
        self._check_dataframe(df)

    def test_nonexisting_country(self, nt: sd.NationalTeamsElo) -> None:
        """Should raise ValueError if the country does not exist."""
        with pytest.raises(ValueError, match="No ratings are available for the given country."):
            _ = nt.read_country_history("Francia")
