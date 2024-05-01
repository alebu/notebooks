import numpy as np
import progressbar
import pandas as pd

def simulate_lending(period_cohort_table, beta_0, beta_age):
    loan_term = 15
    book_age = {}
    defaults = []
    reimbursed = []
    years_on_book = []
    loan_id = 0
    # Iterate over the years
    for y in progressbar.progressbar(period_cohort_table.index.unique()):
        # Originate New Loans
        for i in range(period_cohort_table.loc[y, "originations"]):
            book_age[loan_id] = 0
            loan_id += 1

        # Iterate over active loans and simulate default
        active_loans = [k for k in book_age.keys()]
        for k in active_loans:
            # calculate default Probability based on Age, Period, Cohort
            period = period_cohort_table.loc[y, "period"]
            age = book_age[k]
            cohort = period_cohort_table.loc[y - age, "cohort"]
            default_proba = 1/(1 + np.exp(-(beta_0 + period + cohort + beta_age[age])))
            # Simulate Default
            default = np.random.binomial(1, default_proba)
            if default:
                defaulted_loan = {"loan_id":k, "age":book_age.pop(k), "year":y, "default":1}
                defaults.append(defaulted_loan)
                years_on_book.append(defaulted_loan)
            else:
                years_on_book.append( {"loan_id":k, "age":book_age[k], "year":y, "default":0})
                book_age[k] += 1
                if book_age[k] == loan_term:
                    reimbursed.append({"loan_id":k, "age":book_age.pop(k), "year":y})
    years_on_book_df = pd.DataFrame(years_on_book).rename(columns = {"year":"period"})
    years_on_book_df["cohort"] = years_on_book_df["period"] - years_on_book_df["age"]
    return years_on_book_df
