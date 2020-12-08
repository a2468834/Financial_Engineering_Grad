from   bs4 import BeautifulSoup as BS
import itertools
from   selenium import webdriver
from   selenium.webdriver.common.by import By
from   selenium.webdriver.support.ui import WebDriverWait as WDW
from   selenium.webdriver.support import expected_conditions as EC
import time


class CONST:
    tickers = lambda : ["A", "AAL", "AAP", "AAPL", "ABBV", "ABC", "ABMD", "ABT", "ACN", "ADBE", 
                        "ADI", "ADM", "ADP", "ADS", "ADSK", "AEE", "AEP", "AES", "AFL", "AGN", 
                        "AIG", "AIV", "AIZ", "AJG", "AKAM", "ALB", "ALGN", "ALK", "ALL", "ALLE", 
                        "ALXN", "AMAT", "AMCR", "AMD", "AME", "AMGN", "AMP", "AMT", "AMZN", "ANET", 
                        "ANSS", "ANTM", "AON", "AOS", "APA", "APD", "APH", "APTV", "ARE", "ARNC", 
                        "ATO", "ATVI", "AVB", "AVGO", "AVY", "AWK", "AXP", "AZO", "BA", "BAC", 
                        "BAX", "BBY", "BDX", "BEN", "BF-B", "BIIB", "BK", "BKNG", "BKR", "BLK", 
                        "BLL", "BMY", "BR", "BRK-B", "BSX", "BWA", "BXP", "C", "CAG", "CAH", "CAT", 
                        "CB", "CBOE", "CBRE", "CCI", "CCL", "CDNS", "CDW", "CE", "CERN", "CF", 
                        "CFG", "CHD", "CHRW", "CHTR", "CI", "CINF", "CL", "CLX", "CMA", "CMCSA", 
                        "CME", "CMG", "CMI", "CMS", "CNC", "CNP", "COF", "COG", "COO", "COP", 
                        "COST", "COTY", "CPB", "CPRI", "CPRT", "CRM", "CSCO", "CSX", "CTAS", "CTL", 
                        "CTSH", "CTVA", "CTXS", "CVS", "CVX", "CXO", "D", "DAL", "DD", "DE", "DFS", 
                        "DG", "DGX", "DHI", "DHR", "DIS", "DISCA", "DISCK", "DISH", "DLR", "DLTR", 
                        "DOV", "DOW", "DRE", "DRI", "DTE", "DUK", "DVA", "DVN", "DXC", "EA", 
                        "EBAY", "ECL", "ED", "EFX", "EIX", "EL", "EMN", "EMR", "EOG", "EQIX", 
                        "EQR", "ES", "ESS", "ETFC", "ETN", "ETR", "EVRG", "EW", "EXC", "EXPD", 
                        "EXPE", "EXR", "F", "FANG", "FAST", "FB", "FBHS", "FCX", "FDX", "FE", 
                        "FFIV", "FIS", "FISV", "FITB", "FLIR", "FLS", "FLT", "FMC", "FOX", "FOXA", 
                        "FRC", "FRT", "FTI", "FTNT", "FTV", "GD", "GE", "GILD", "GIS", "GL", 
                        "GLW", "GM", "GOOG", "GOOGL", "GPC", "GPN", "GPS", "GRMN", "GS", "GWW", 
                        "HAL", "HAS", "HBAN", "HBI", "HCA", "HD", "HES", "HFC", "HIG", "HII", 
                        "HLT", "HOG", "HOLX", "HON", "HP", "HPE", "HPQ", "HRB", "HRL", "HSIC", 
                        "HST", "HSY", "HUM", "IBM", "ICE", "IDXX", "IEX", "IFF", "ILMN", "INCY", 
                        "INFO", "INTC", "INTU", "IP", "IPG", "IPGP", "IQV", "IR", "IRM", "ISRG", 
                        "IT", "ITW", "IVZ", "J", "JBHT", "JCI", "JKHY", "JNJ", "JNPR", "JPM", 
                        "JWN", "K", "KEY", "KEYS", "KHC", "KIM", "KLAC", "KMB", "KMI", "KMX", "KO", 
                        "KR", "KSS", "KSU", "L", "LB", "LDOS", "LEG", "LEN", "LH", "LHX", "LIN", 
                        "LKQ", "LLY", "LMT", "LNC", "LNT", "LOW", "LRCX", "LUV", "LVS", "LW", 
                        "LYB", "LYV", "M", "MA", "MAA", "MAR", "MAS", "MCD", "MCHP", "MCK", "MCO", 
                        "MDLZ", "MDT", "MET", "MGM", "MHK", "MKC", "MKTX", "MLM", "MMC", "MMM", 
                        "MNST", "MO", "MOS", "MPC", "MRK", "MRO", "MS", "MSCI", "MSFT", "MSI", 
                        "MTB", "MTD", "MU", "MXIM", "MYL", "NBL", "NCLH", "NDAQ", "NEE", "NEM", 
                        "NFLX", "NI", "NKE", "NLOK", "NLSN", "NOC", "NOV", "NOW", "NRG", "NSC", 
                        "NTAP", "NTRS", "NUE", "NVDA", "NVR", "NWL", "NWS", "NWSA", "O", "ODFL", 
                        "OKE", "OMC", "ORCL", "ORLY", "OXY", "PAYC", "PAYX", "PBCT", "PCAR", 
                        "PEAK", "PEG", "PEP", "PFE", "PFG", "PG", "PGR", "PH", "PHM", "PKG", "PKI", 
                        "PLD", "PM", "PNC", "PNR", "PNW", "PPG", "PPL", "PRGO", "PRU", "PSA", 
                        "PSX", "PVH", "PWR", "PXD", "PYPL", "QCOM", "QRVO", "RCL", "RE", "REG", 
                        "REGN", "RF", "RHI", "RJF", "RL", "RMD", "ROK", "ROL", "ROP", "ROST", 
                        "RSG", "RTX", "SBAC", "SBUX", "SCHW", "SEE", "SHW", "SIVB", "SJM", "SLB", 
                        "SLG", "SNA", "SNPS", "SO", "SPG", "SPGI", "SRE", "STE", "STT", "STX", 
                        "STZ", "SWK", "SWKS", "SYF", "SYK", "SYY", "T", "TAP", "TDG", "TEL", "TFC", 
                        "TFX", "TGT", "TIF", "TJX", "TMO", "TMUS", "TPR", "TROW", "TRV", "TSCO", 
                        "TSN", "TT", "TTWO", "TWTR", "TXN", "TXT", "UA", "UAA", "UAL", "UDR", 
                        "UHS", "ULTA", "UNH", "UNM", "UNP", "UPS", "URI", "USB", "V", "VAR", "VFC", 
                        "VIAC", "VLO", "VMC", "VNO", "VRSK", "VRSN", "VRTX", "VTR", "VZ", "WAB", 
                        "WAT", "WBA", "WDC", "WEC", "WELL", "WFC", "WHR", "WLTW", "WM", "WMB", 
                        "WMT", "WRB", "WRK", "WU", "WY", "WYNN", "XEL", "XLNX", "XOM", "XRAY", 
                        "XRX", "XYL", "YUM", "ZBH", "ZBRA", "ZION", "ZTS", "BIV", "LQD", "MUB", 
                        "TLT", "VB", "VNQ", "VOO", "VEA", "VWO", "IAU", "1101.TW", "1102.TW", 
                        "1216.TW", "1301.TW", "1303.TW", "1326.TW", "1402.TW", "2002.TW", 
                        "2105.TW", "2207.TW", "2227.TW", "2301.TW", "2303.TW", "2308.TW", 
                        "2317.TW", "2327.TW", "2330.TW", "2352.TW", "2357.TW", "2382.TW", 
                        "2395.TW", "2408.TW", "2412.TW", "2454.TW", "2474.TW", "2609.TW", 
                        "2610.TW", "2633.TW", "2801.TW", "2823.TW", "2880.TW", "2881.TW", 
                        "2882.TW", "2883.TW", "2884.TW", "2885.TW", "2886.TW", "2887.TW", 
                        "2888.TW", "2890.TW", "2891.TW", "2892.TW", "2912.TW", "3008.TW", 
                        "3045.TW", "3711.TW", "4904.TW", "4938.TW", "5871.TW", "5876.TW", 
                        "5880.TW", "6505.TW", "6669.TW", "9910.TW", "0050.TW", "BTC", "ETH", 
                        "00720B.TW", "00725B.TW", "00740B.TW", "00751B.TW", "00862B.TW", "5274.TW", 
                        "6274.TW", "6470.TW", "3491.TW", "3105.TW", "5439.TW", "XAU"]
    TW_tickers = lambda : ['1101.TW', '1102.TW', '1216.TW', '1301.TW', '1303.TW', '1326.TW', 
                           '1402.TW', '2002.TW', '2105.TW', '2207.TW', '2227.TW', '2301.TW', 
                           '2303.TW', '2308.TW', '2317.TW', '2327.TW', '2330.TW', '2352.TW', 
                           '2357.TW', '2382.TW', '2395.TW', '2408.TW', '2412.TW', '2454.TW', 
                           '2474.TW', '2609.TW', '2610.TW', '2633.TW', '2801.TW', '2823.TW', 
                           '2880.TW', '2881.TW', '2882.TW', '2883.TW', '2884.TW', '2885.TW', 
                           '2886.TW', '2887.TW', '2888.TW', '2890.TW', '2891.TW', '2892.TW', 
                           '2912.TW', '3008.TW', '3045.TW', '3711.TW', '4904.TW', '4938.TW', 
                           '5871.TW', '5876.TW', '5880.TW', '6505.TW', '6669.TW', '9910.TW', 
                           '0050.TW']
    other_etf = lambda : ["BIV", "LQD", "MUB", "TLT", "VB", "VNQ", "VOO", "VEA", "VWO", "IAU"]

def login(driver):
    driver.get('http://qffers.qf.nthu.edu.tw:8002/login') # Open webpage
    
    email  = driver.find_element_by_name("username")
    passwd = driver.find_element_by_name("password")
    
    # Enter login info
    email.send_keys('a2468834@gmail.com')
    passwd.send_keys('Ewc9Nmp9NUJ2')
    passwd.submit()
    
    # Wait for the completion of login
    WDW(driver, 10).until(EC.presence_of_element_located((By.XPATH, '//*[@href="/logout"]')))


def createStrategy(driver, ticker_comb):
    driver.get('http://qffers.qf.nthu.edu.tw:8002/create_strategy')
    
    strategy_name = str(int(time.time()))
    name_form_group = driver.find_element_by_name("strategy_name")
    name_form_group.send_keys(strategy_name)
    
    add_ticker = driver.find_element_by_id("add")
    add_ticker.click()
    add_ticker.click()
    
    ticker_form_groups = driver.find_elements_by_id("asset_ticker")
    for form_group, index in zip(ticker_form_groups, range(len(ticker_form_groups))):
        form_group.clear()
        form_group.send_keys(ticker_comb[index])
    
    submit_btn = driver.find_element_by_xpath("/html/body/div/div[2]/div/div/div[1]/main/div/div/div/div/div[2]/form/div[3]/input[2]")
    submit_btn.click()
    driver.switch_to.alert.accept()
    
    # Wait for the completion of creating strategy
    try:
        WDW(driver, 600).until(EC.presence_of_element_located((By.XPATH, '/html/body/nav/div[@class="alert alert-success"]')))
    except:
        print("Creating strategy is time out.")
        print("\n[Information]")
        print(strategy_name)
        print("Portfolio = {} + {} + {}\n".format(ticker_comb[0], ticker_comb[1], ticker_comb[2]))


if __name__ == '__main__':
    all_ticker_comb = list(itertools.combinations(CONST.other_etf(), 3))
    driver = webdriver.Firefox()
    
    login(driver)
    for ticker_comb in all_ticker_comb:
        start = time.time()
        createStrategy(driver, ticker_comb)
        print("Elapsed time: {:.2f}".format(time.time()-start))
        time.sleep(1)
    
    driver.close() # Close the current window
