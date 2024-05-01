def prepare_period_cohort_table(
        uk_cpi, uk_unemp, 
        unnormalised_beta_unemp_period, unnormalised_beta_inflation_period,
        unnormalised_beta_unemp_cohort, unnormalised_beta_inflation_cohort
    ):
    df = uk_unemp.groupby(uk_unemp.index.year).mean().reset_index().rename(columns = {"index":"year", 0:"unemp"}).merge(
        uk_cpi.groupby(uk_cpi.index.year).mean().reset_index().rename(columns = {"index":"year", 0:"CPI"})
    )
    df["inflation"] = df["CPI"].diff()/df["CPI"]
    df["unemp"] = df["unemp"]/100

    df = df.set_index("year")
    df = df[df.index > 1990].drop("CPI", axis = 1)
    df["period"] = df["unemp"]*unnormalised_beta_unemp_period + df["inflation"]*unnormalised_beta_inflation_period
    df["cohort"] = df["unemp"]*unnormalised_beta_unemp_cohort + df["inflation"]*unnormalised_beta_inflation_cohort
    df["originations"] = 5000
    return df


def preprocess_data_for_apc_analysis(yob, period_cohort_table, test_cutoff_year):
    apc_df = (yob[yob.period < test_cutoff_year].groupby(["age", "period", "cohort"])
            ["default"]
            .agg(("size", "sum")))

    apc_df_test = (yob[yob.period >= test_cutoff_year].groupby(["age", "period", "cohort"])
                ["default"]
                .agg(("size", "sum")))

    apc_df = apc_df.reset_index().merge(
        period_cohort_table[["unemp", "inflation"]].reset_index(),
        left_on = "period", 
        right_on = "year"
    ).drop("year", axis = 1).set_index(["age", "period", "cohort"])

    apc_df_test = apc_df_test.reset_index().merge(
        period_cohort_table[["unemp", "inflation"]].reset_index(),
        left_on = "period", 
        right_on = "year"
    ).drop("year", axis = 1).set_index(["age", "period", "cohort"])
    return apc_df, apc_df_test