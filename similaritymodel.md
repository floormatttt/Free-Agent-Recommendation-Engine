Similar Player Model
    Inputs
        Player vectors
    Outputs
        List of the 10 closest players and their contracts
        Weighted average of the 10 contracts by distance
    Process
        Mahalanobis distance between two player vectors
        4:1 or 5:1 ratio between parallel axis length and perpendicular axis length
        Find top 10 by distance to compare contracts
