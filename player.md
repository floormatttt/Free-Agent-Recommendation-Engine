Player Profile Model: 
    Input:
        Take in all the player data
    Output:
        Player vector with 12 or so categories
    Process:
        Playtype data: 
            3 years of player data for each player
                Rookies: just use that one year
                Sophomores: use two years
                Not enough of a playstyle? Assume baseline of 0.5% frequency and 0.9 PPP
            Experience-adjusted time decay
                Current season weights peak at beginning of career and end of career
                Current season weights bottom out at prime of career
                Descending weights for current season, season before, and 2 seasons before
                Missed a season with injury? ignore it for now
            Apply experience-adjusted time decay to both FREQ%, PPP, and total possessions
            Calculate:
                Z-score of player’s FREQ% - z-score of 0 FREQ%
                Z-score of player’s PPP - z-score of 0 PPP
            Generate playtype score using geometric mean of PPP-score and FREQ%-score
        Defense:
            rim_block = (BLK%_z^0.25 × DFG%_z_inverted^0.30 × Diff%_z_inverted^0.35 × DFGA_z^0.10)
            reb_block = (DREB%_z × OREB%_z × def_box_outs_pg_z) ^ (1/3)
            Adjust everything using experience-adjusted time decay too 

    Contribution to final recommendation engine:
        This model balances both the frequencies that a player runs certain types of plays with their effectiveness, building a complete player profile that allows teams to see where that player fits into their team. (fit model)
        Additionally, the player vector can be used to compare historically to other player vectors too, noting similarities in salary cap percentages
    Notes:
        Injuries? 