import numpy as np
from scipy import stats


def t_test_for_attribution():
    print('execute t-test for attribute')
    # pvalue is:  0.0464668057793
    mean1 = 87.64
    mean2 = 83.32

    std1 = 0.26
    std2 = 1.34

    nobs1 = 2
    nobs2 = 2

    (statistic, pvalue) = stats.ttest_ind_from_stats(mean1=mean1, std1=std1, nobs1=nobs1, mean2=mean2,
                                                     std2=std2, nobs2=nobs2)

    print "t statistic is: ", statistic
    print "pvalue is: ", pvalue


def t_test_for_clstm():
    print('execute t-test for bd-lstm')
    # pvalue is:  0.123142952777
    mean1 = 87.64
    mean2 = 82.30

    std1 = 0.26
    std2 = 3.27

    nobs1 = 2
    nobs2 = 2

    (statistic, pvalue) = stats.ttest_ind_from_stats(mean1=mean1, std1=std1, nobs1=nobs1, mean2=mean2,
                                                     std2=std2, nobs2=nobs2)

    print "t statistic is: ", statistic
    print "pvalue is: ", pvalue


def t_test_for_finetuneed():
    print('execute t-test for finetuned')
    # pvalue is:  0.123142952777
    mean1 = 87.64
    mean2 = 80.33

    std1 = 0.26
    std2 = 6.14

    nobs1 = 2
    nobs2 = 2

    (statistic, pvalue) = stats.ttest_ind_from_stats(mean1=mean1, std1=std1, nobs1=nobs1, mean2=mean2,
                                                     std2=std2, nobs2=nobs2)

    print "t statistic is: ", statistic
    print "pvalue is: ", pvalue


def t_test_for_intraloss():
    print('execute t-test for center loss(intra-loss)')
    # pvalue is:  0.00111352494053
    mean1 = 87.64
    mean2 = 79.55

    std1 = 0.26
    std2 = 0.28

    nobs1 = 2
    nobs2 = 2

    (statistic, pvalue) = stats.ttest_ind_from_stats(mean1=mean1, std1=std1, nobs1=nobs1, mean2=mean2,
                                                     std2=std2, nobs2=nobs2)

    print "t statistic is: ", statistic
    print "pvalue is: ", pvalue


if __name__ == '__main__':
    t_test_for_attribution()
    t_test_for_intraloss()
    t_test_for_clstm()
    t_test_for_finetuneed()