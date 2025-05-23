from numpy import sqrt, log
from scipy.stats import norm



def BSfwd(F,X,Z,sigma,T,CallF):

    # BSfwd(F,X,Z,sigma,T,CallF)
    # Compute Black's option premium
    # F = forward price
    # X = strike
    # Z = discount
    # sigma = volatility of forward
    # T = time-to-maturity
    # CallF = flag for call. = 1 for calls and =2 for puts

    if CallF==1:
        d1    = (log(F/X)+(sigma**2/2)*T)/(sigma*sqrt(T))
        d2    = d1-sigma*sqrt(T)
        Price = Z*(F*norm.cdf(d1)-X*norm.cdf(d2))

    elif CallF==2:
        d1    = (log(F/X)+(sigma**2/2)*T)/(sigma*sqrt(T))
        d2    = d1-sigma*sqrt(T)
        Price = Z*(-F*norm.cdf(-d1)+X*norm.cdf(-d2))

    return Price
