1. **id**: Unique identifier for each loan.
2. **member_id**: Unique identifier for each borrower/member.
3. **loan_amnt**: The amount of money requested in the loan application.
4. **funded_amnt**: The total amount of the loan funded.
5. **funded_amnt_inv**: The total amount committed by investors for that loan at that point in time.
6. **term**: The length of the loan term (e.g., 36 months, 60 months).
7. **int_rate**: The interest rate on the loan.
8. **installment**: The monthly payment owed by the borrower.
9. **grade**: Loan grade assigned by the lending institution.
10. **sub_grade**: Further classification within loan grade.
11. **emp_title**: Employment title of the borrower.
12. **emp_length**: Length of employment in years.
13. **home_ownership**: Type of home ownership (e.g., own, mortgage, rent).
14. **annual_inc**: Annual income reported by the borrower.
15. **verification_status**: Indicates if income was verified by the lending institution.
16. **issue_d**: Date when the loan was issued.
17. **loan_status**: Current status of the loan (e.g., fully paid, charged off).
18. **pymnt_plan**: Indicates if a payment plan has been put in place.
19. **url**: URL for the loan on the lending platform.
20. **desc**: Description provided by the borrower.
21. **purpose**: Purpose of the loan.
22. **title**: Loan title provided by the borrower.
23. **zip_code**: Borrower's zip code.
24. **addr_state**: Borrower's state.
25. **dti**: Debt-to-income ratio.
26. **delinq_2yrs**: Number of 30+ days past-due incidences of delinquency in the borrower's credit file for the past 2 years.
27. **earliest_cr_line**: The date when the borrower's earliest reported credit line was opened.
28. **inq_last_6mths**: Number of inquiries in the last 6 months.
29. **mths_since_last_delinq**: The number of months since the borrower's last delinquency.
30. **mths_since_last_record**: The number of months since the last public record.
31. **open_acc**: The number of open credit lines in the borrower's credit file.
32. **pub_rec**: Number of derogatory public records.
33. **revol_bal**: Total credit revolving balance.
34. **revol_util**: Revolving line utilization rate (percentage of available credit used).
35. **total_acc**: The total number of credit lines currently in the borrower's credit file.
36. **initial_list_status**: The initial listing status of the loan (e.g., W, F).
37. **out_prncp**: Remaining outstanding principal for total amount funded.
38. **out_prncp_inv**: Remaining outstanding principal for portion of total amount funded by investors.
39. **total_pymnt**: Payments received to date for total amount funded.
40. **total_pymnt_inv**: Payments received to date for portion of total amount funded by investors.
41. **total_rec_prncp**: Principal received to date.
42. **total_rec_int**: Interest received to date.
43. **total_rec_late_fee**: Late fees received to date.
44. **recoveries**: Post charge-off gross recovery.
45. **collection_recovery_fee**: Post charge-off collection fee.
46. **last_pymnt_d**: Last month payment was received.
47. **last_pymnt_amnt**: Last total payment amount received.
48. **next_pymnt_d**: Next scheduled payment date.
49. **last_credit_pull_d**: The most recent month LC pulled credit for this loan.
50. **collections_12_mths_ex_med**: Number of collections in 12 months excluding medical collections.
51. **mths_since_last_major_derog**: Months since most recent 90-day or worse rating.
52. **policy_code**: publicly available policy_code=1.
53. **application_type**: Indicates whether the loan is an individual application or a joint application with two co-borrowers.
54. **annual_inc_joint**: The combined annual income provided by the co-borrowers during registration.
55. **dti_joint**: A ratio calculated using the co-borrowers' total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the co-borrowers' combined monthly income.
56. **verification_status_joint**: Indicates if the co-borrowers' joint income was verified by the lending institution.
57. **acc_now_delinq**: The number of accounts on which the borrower is now delinquent.
58. **tot_coll_amt**: Total collection amounts ever owed.
59. **tot_cur_bal**: Total current balance of all accounts.
60. **open_acc_6m**: Number of open trades in the last 6 months.
61. **open_il_6m**: Number of currently active installment trades.
62. **open_il_12m**: Number of installment accounts opened in past 12 months.
63. **open_il_24m**: Number of installment accounts opened in past 24 months.
64. **mths_since_rcnt_il**: Months since most recent installment accounts opened.
65. **total_bal_il**: Total current balance of all installment accounts.
66. **il_util**: Ratio of total current balance to high credit/credit limit on all install acct.
67. **open_rv_12m**: Number of revolving trades opened in the last 12 months.
68. **open_rv_24m**: Number of revolving trades opened in the last 24 months.
69. **max_bal_bc**: Maximum current balance owed on all revolving accounts.
70. **all_util**: Balance to credit limit on all trades.
71. **total_rev_hi_lim**: Total revolving high credit/credit limit.
72. **inq_fi**: Number of personal finance inquiries.
73. **total_cu_tl**: Number of finance trades.
74. **inq_last_12m**: Number of credit inquiries in the last 12 months.
