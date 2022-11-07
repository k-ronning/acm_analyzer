import datetime
import dateutil
import platform
import isoweek
import datetime
import math
import re
import numpy
import scipy.optimize
import scipy.interpolate
import scipy.integrate
import pandas
import matplotlib.pyplot as plt
import matplotlib.dates
import matplotlib.ticker
import matplotlib.patches
import os
import sys

def init_latex():
    global Y2020_COLOR_OFFSET
    if False:
        plt.style.use('seaborn')
        Y2020_COLOR_OFFSET = 1
    elif False:
        plt.style.use('ggplot')
        Y2020_COLOR_OFFSET = 0
    else:
        plt.style.use('bmh')
        Y2020_COLOR_OFFSET = 0

    # Documentation:
    # https://matplotlib.org/stable/tutorials/introductory/customizing.html#the-default-matplotlibrc-file
    plt.rcParams.update({
        "text.usetex": False, # True if platform.system() != "Windows" else False,
        #"text.latex.preamble": [
        #    r"\usepackage{graphics}",
        #    r"\usepackage{tgtermes}",
        #],
        # "font.family": "tgtermes" if platform.system() != "Windows" else "Times New Roman",
        "font.family": "Times New Roman",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "cm",
        "mathtext.rm": "serif",
        #"mathtext.fallback": "cm",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 9,
        "font.size": 9,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "grid.linewidth": 0.3,
        "grid.alpha": 1.0,
    })

# how to embed Matplitlib in Latex:
# https://jwalton.info/Embed-Publication-Matplotlib-Latex/

DAYS_IN_YEAR_EXACT = 365.24
WEEKS_IN_YEAR_EXACT = DAYS_IN_YEAR_EXACT / 7
BASELINE_START_DATE = datetime.date(2008, 1, 1)
BASELINE_CUTOFF_DATE = datetime.date(2020, 1, 1)
BASELINE_INTERVAL = datetime.timedelta(days=365*3)
T0_DATE = datetime.date(2020, 1, 1)
T0_DATETIME64 = numpy.datetime64("2020-01-01")
YEAR_WEEK_RE = re.compile(r'"(\d\d\d\d)W(\d\d)\*?"')
PAPER_WIDTH_PT = 483.7
PAPER_WIDTH_IN = PAPER_WIDTH_PT / 72.27
COLUMN_WIDTH_PT = 234.0
COLUMN_WIDTH_IN = COLUMN_WIDTH_PT / 72.27
DEFAULT_PAD_INCHES = 0.01
DEFAULT_LINEWIDTH = 0.5
DEFAULT_DPI = 300
EXCESS_MORTALITY_FILL_COLOR = "#ff2222"
LOW_MORTALITY_FILL_COLOR = "C6"
BASELINE_PLOT_HEIGHT = 1.1
BLUE_COLOR = "#3366ee"
ACM_CUSTOM_YLIM = (850, 1300)
ESTIMATION_PAST_YEARS=10
BASELINE_PLOT_START_DATE = BASELINE_START_DATE
BASELINE_PLOT_END_DATE = datetime.date(2022, 12, 31)

def get_date_from_isoweek(year, week):
    return isoweek.Week(year, week).thursday()

def set_size(width, fraction=1, subplots=(1, 1), height_in_override=None):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    if height_in_override is None:
        fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])
    else:
        fig_height_in = height_in_override

    return (fig_width_in, fig_height_in)

def save_fig(fig, fig_file_name):
    pdf_file_name = fig_file_name + ".pdf"
    fig.savefig(pdf_file_name,
                dpi = DEFAULT_DPI,
                # "bbox_inches": "tight",
                pad_inches = DEFAULT_PAD_INCHES,
                transparent = True)
    print("Wrote %s" % (pdf_file_name,))
    png_file_name = fig_file_name + ".png"
    fig.savefig(png_file_name,
                dpi = DEFAULT_DPI, 
                # "bbox_inches": "tight",
                pad_inches = DEFAULT_PAD_INCHES,
                transparent = True,
                facecolor = "white")
    print("Wrote %s" % (png_file_name,))


class WeekNumberLocator(matplotlib.dates.RRuleLocator):
    """
    Make ticks on occurrences of week numbers.
    """
    def __init__(self, byweekno=None, byweekday=dateutil.rrule.TH, interval=1, tz=None):
        """
        Mark every ISO 8601 week number in *byweekno*; *byweekno* can be an int or
        sequence.  Default is ``range(1,53)``, i.e. every week.

        *interval* is the interval between each iteration.  For
        example, if ``interval=2``, mark every second occurrence.
        """
        if byweekno is None:
            byweekno = range(1, 53)
        elif isinstance(byweekno, numpy.ndarray):
            # This fixes a bug in dateutil <= 2.3 which prevents the use of
            # numpy arrays in (among other things) the bymonthday, byweekday
            # and bymonth parameters.
            byweekno = [x.item() for x in byweekno.astype(int)]

        rule = matplotlib.dates.rrulewrapper(dateutil.rrule.WEEKLY, byweekno=byweekno, byweekday=byweekday,
                                             interval=interval, **self.hms0d)
        matplotlib.dates.RRuleLocator.__init__(self, rule, tz)

class WeekNumberFormatter(matplotlib.dates.DateFormatter):
    def __init__(self, tz=None):
        matplotlib.dates.DateFormatter.__init__(self, "%V", tz)
    def __call__(self, x, pos=0):
        return matplotlib.dates.DateFormatter.__call__(self, x, pos).lstrip("0")

# Parse CSV file downloaded from:
# https://pxnet2.stat.fi/PXWeb/pxweb/fi/Kokeelliset_tilastot/Kokeelliset_tilastot__vamuu_koke/koeti_vamuu_pxt_12ng.px/
#
# Choices for obtaining CSV:
#   Alue = KOKO MAA
#   Viikko = Valitse kaikki
#   Ikä = Valitse kaikki
#   Sukupuoli = Valitse kaikki
# 
# Click Lataa taulukko > Lataa puolipiste-eroteltu csv-tiedosto (otsikollinen)
#
def parse_finland_acm_csv(csv_file, trim_weeks_from_end):
    out_tuples = []
    acm_by_category = {}
    all_lines = list(csv_file.readlines())
    heading_line = all_lines[2]
    heading_cells = heading_line.strip().split(";")
    assert heading_cells[0] == '"Alue"'
    assert heading_cells[1] == '"Viikko"'
    assert heading_cells[2] == '"Yhteensä Yhteensä Kuolleet"'
    assert heading_cells[3] == '"Yhteensä Miehet Kuolleet"'
    assert heading_cells[4] == '"Yhteensä Naiset Kuolleet"'
    age_categories = ((0, 4, []),
                      (5, 9, ["5-19"]),
                      (10, 14, ["5-19"]),
                      (15, 19, ["5-19"]),
                      (20, 24, ["20-39", "20-64"]),
                      (25, 29, ["20-39", "20-64"]),
                      (30, 34, ["20-39", "20-64"]),
                      (35, 39, ["20-39", "20-64"]),
                      (40, 44, ["40-64", "20-64"]),
                      (45, 49, ["40-64", "20-64"]),
                      (50, 54, ["40-64", "20-64"]),
                      (55, 59, ["40-64", "20-64"]),
                      (60, 64, ["40-64", "20-64"]),
                      (65, 69, ["65-"]),
                      (70, 74, ["65-"]),
                      (75, 79, ["65-"]),
                      (80, 84, ["65-"]),
                      (85, 89, ["65-"]),
                      (90, None, ["65-"]))
    column_idx = 5
    for age_start, age_end, meta_categories in age_categories:
        age_category_str = str(age_start) + " -" 
        if age_end:
            age_category_str += " " + str(age_end)
        assert heading_cells[column_idx] == '"%s Yhteensä Kuolleet"' % (age_category_str,)
        assert heading_cells[column_idx+1] == '"%s Miehet Kuolleet"' % (age_category_str,)
        assert heading_cells[column_idx+2] == '"%s Naiset Kuolleet"' % (age_category_str,)
        category_key = str(age_start) + "-" 
        if age_end:
            category_key += str(age_end)
        acm_by_category[category_key] = []
        acm_by_category[category_key + "_male"] = []
        acm_by_category[category_key + "_female"] = []
        for meta_category in meta_categories:
            if meta_category not in acm_by_category:
                acm_by_category[meta_category] = []
                acm_by_category[meta_category + "_male"] = []
                acm_by_category[meta_category + "_female"] = []
        column_idx += 3
    year = None
    week = None
    for line in all_lines[3:]:
        line = line.strip()
        if not line:
            break
        cells = line.strip().split(";")
        assert cells[0] == '"KOKO MAA"'
        year_week_match = YEAR_WEEK_RE.match(cells[1])
        assert year_week_match, "No match: %s" % (cells[1],)
        year = int(year_week_match.group(1))
        week = int(year_week_match.group(2))
        deaths = int(cells[2])
        male_deaths = int(cells[3])
        female_deaths = int(cells[4])
        begin_date = get_date_from_isoweek(year, week)
        # sanity check for data
        assert year >= 1900 and year < 2100
        assert week >= 1 and week <= 53
        assert begin_date >= datetime.date(1900, 1, 1) and begin_date < datetime.date(2100, 1, 1)
        assert deaths > 0 and deaths < 10000
        out_tuples.append((begin_date, deaths))
        
        category_deaths_sum = 0
        male_deaths_sum = 0
        female_deaths_sum = 0
        column_idx = 5
        output_meta_categories = dict()
        for age_start, age_end, meta_categories in age_categories:
            category_key = str(age_start) + "-" 
            if age_end:
                category_key += str(age_end)
            age_deaths = int(cells[column_idx])
            age_male_deaths = int(cells[column_idx+1])
            age_female_deaths = int(cells[column_idx+2])
            assert age_deaths == age_male_deaths + age_female_deaths
            category_deaths_sum += age_deaths
            acm_by_category[category_key].append((begin_date, age_deaths))
            male_deaths_sum += age_male_deaths
            acm_by_category[category_key + "_male"].append((begin_date, age_male_deaths))
            female_deaths_sum += age_female_deaths
            acm_by_category[category_key + "_female"].append((begin_date, age_female_deaths))
            for meta_category in meta_categories:
                if meta_category not in output_meta_categories:
                    output_meta_categories[meta_category] = 0
                    output_meta_categories[meta_category + "_male"] = 0
                    output_meta_categories[meta_category + "_female"] = 0
                output_meta_categories[meta_category] += age_deaths
                output_meta_categories[meta_category + "_male"] += age_male_deaths
                output_meta_categories[meta_category + "_female"] += age_female_deaths
            column_idx += 3
        for meta_category, meta_category_sum in output_meta_categories.items():
            acm_by_category[meta_category].append((begin_date, meta_category_sum))
        assert category_deaths_sum == deaths
        assert male_deaths_sum == male_deaths
        assert female_deaths_sum == female_deaths
    # trim data from end because it is not final
    return out_tuples[:-trim_weeks_from_end], dict(((k, v[:-trim_weeks_from_end]) for k, v in acm_by_category.items()))

THL_YEAR_WEEK_RE = re.compile(r"Vuosi (\d\d\d\d) Viikko (\d\d)")

# Parse CSV file downloaded from:
# https://sampo.thl.fi/pivot/prod/fi/epirapo/covid19case/fact_epirapo_covid19case?row=dateweek20200101-509030&column=measure-444833.445356.492118.&fo=1
# 
# Click: Vie taulukko > CSV-tiedostoon
#
def parse_finland_thl_covid_data_csv(csv_file):
    out_tuples = []
    all_lines = list(csv_file.readlines())
    heading_line = all_lines[0].strip()
    assert heading_line == "Mittari;Aika;val"
    cur_year = None
    cur_week = None
    cur_begin_date = None
    cur_covid_cases = None
    cur_covid_tests = None
    for line_idx, line in enumerate(all_lines[1:]):
        line_no = line_idx+2
        line = line.strip()
        if not line or ";Kaikki ajat;" in line:
            break
        cells = line.split(";")
        line_type = cells[0]
        year_week_match = THL_YEAR_WEEK_RE.fullmatch(cells[1])
        assert year_week_match is not None
        year = int(year_week_match.group(1))
        week = int(year_week_match.group(2))
        begin_date = get_date_from_isoweek(year, week)
        if line_type == "Tapausten lukumäärä":
            cur_covid_cases = int(cells[2] or 0)
            assert cur_year == None
            assert cur_week == None
            assert cur_begin_date == None
            cur_year = year
            cur_week = week
            cur_begin_date = begin_date
        elif line_type == "Testausmäärä":
            assert cur_year == year, "Year %s != %s on line %d" % (cur_year, year, line_no)
            assert cur_week == week
            assert cur_begin_date == begin_date
            cur_covid_tests = int(cells[2] or 0)
        elif line_type == "Kuolemantapausten lukumäärä":
            assert cur_year == year
            assert cur_week == week
            assert cur_begin_date == begin_date
            assert cur_covid_cases is not None
            assert cur_covid_tests is not None
            covid_deaths = int(cells[2] or 0)
            out_tuples.append((begin_date, covid_deaths, cur_covid_cases, cur_covid_tests))
            cur_year = None
            cur_week = None
            cur_begin_date = None
            cur_covid_cases = None
            cur_covid_tests = None
        else:
            assert False, "Unkown line type: %s" % (line_type,)
    return out_tuples

EUROMOMO_YEAR_WEEK_RE = re.compile(r"(\d\d\d\d)-(\d\d)")

# Parse CSV file downloaded from:
# https://www.euromomo.eu/graphs-and-maps/
# 
# Click: Z-scores by country > All ages, Countries Finland > Download data
#
def parse_euromomo_zscores_csv(csv_file):
    out_tuples = []
    all_lines = list(csv_file.readlines())
    heading_line = all_lines[0].strip()
    assert heading_line == "country;group;week;zscore"
    cur_begin_date = None
    cur_item = None
    for line_idx, line in enumerate(all_lines[1:]):
        line_no = line_idx+2
        line = line.strip()
        if not line:
            break
        cells = line.split(";")
        assert len(cells) == 4
        country = cells[0]
        assert country
        line_type = cells[1]
        if line_type != "Total":
            continue
        year_week_match = EUROMOMO_YEAR_WEEK_RE.fullmatch(cells[2])
        assert year_week_match is not None
        year = int(year_week_match.group(1))
        week = int(year_week_match.group(2))
        begin_date = get_date_from_isoweek(year, week)
        if cells[3]:
            z_score = float(cells[3])
            if begin_date != cur_begin_date:
                if cur_item:
                    out_tuples.append((cur_begin_date, cur_item))
                cur_item = dict()
                cur_begin_date = begin_date
            cur_item[country] = z_score
    if cur_item:
        out_tuples.append((cur_begin_date, cur_item))
    return out_tuples

# Parse CSV file downloaded from:
# https://pxnet2.stat.fi/PXWeb/pxweb/fi/StatFin/StatFin__vrm__vaerak/statfin_vaerak_pxt_11rd.px/
# 
# Choices for obtaining CSV:
#   Vuosi = Valitse kaikki
#   Sukupuoli = Valitse kaikki
#   Ikä = Valitse kaikki
# 
# Click: Lataa taulukko > Lataa puolipiste-eroteltu csv-tiedosto (otsikollinen)
#
def parse_finland_tilastokeskus_population_csv(csv_file):
    population_by_year_and_age = {}
    all_lines = list(csv_file.readlines())
    heading_line = all_lines[2].strip()
    heading_cells = heading_line.split(";")
    assert heading_cells[0] == '"Vuosi"'
    assert heading_cells[1].startswith('"Yhteensä Yhteensä Väestö')
    max_age = 112
    for age in range(0, max_age+1):
        assert heading_cells[2+age].startswith('"Yhteensä ' + str(age) + ' Väestö')
        assert heading_cells[2+max_age+2+age].startswith('"Miehet ' + str(age) + ' Väestö')
        assert heading_cells[2+2*(max_age+2)+age].startswith('"Naiset ' + str(age) + ' Väestö')
    for line_idx, line in enumerate(all_lines[3:]):
        cells = line.strip().split(";")
        year = int(cells[0].strip('"'))
        assert 1900 <= year <= 2100
        year_total_population = int(cells[1])
        year_population_sum = 0
        assert year not in population_by_year_and_age
        population_by_year_and_age[year] = {
            "year_total": year_total_population,
        }
        for age in range(0, max_age+1):
            age_population = int(cells[2+age].strip('"'))
            assert 0 <= age_population < 1000000
            age_male_population = int(cells[2+max_age+2+age].strip('"'))
            assert 0 <= age_male_population < 1000000
            age_female_population = int(cells[2+2*(max_age+2)+age].strip('"'))
            assert 0 <= age_female_population < 1000000
            assert age_male_population + age_female_population == age_population
            assert age not in population_by_year_and_age[year]
            population_by_year_and_age[year][age] = age_population, age_male_population, age_female_population
            year_population_sum += age_population
        assert year_total_population == year_population_sum
    return population_by_year_and_age
    
# Parse CSV file downloaded from:
# https://pxnet2.stat.fi/PXWeb/pxweb/fi/StatFin/StatFin__vrm__vaenn/statfin_vaenn_pxt_139e.px/table/tableViewLayout1/
# 
# Choices for obtaining CSV:
#   Tiedot = Kuolleet
#   Vuosi = 2007-2030
#   Sukupuoli = Yhteensä
#   Väestöennuste = Valitse kaikki
# 
# Click: Lataa taulukko > Lataa puolipiste-eroteltu csv-tiedosto (otsikollinen)
#
def parse_finland_population_forecast_csv(csv_file):
    all_lines = list(csv_file.readlines())
    heading_line = all_lines[2].strip()
    heading_cells = heading_line.split(";")
    assert heading_cells[0] == '"Vuosi"'
    assert heading_cells[1] == '"Tiedot"'
    assert heading_cells[2] == '"Yhteensä Todelliset tiedot"'
    years = (2021, 2019, 2018, 2015, 2012, 2009, 2007)
    for index, year in enumerate(years):
        assert heading_cells[3+index] == '"Yhteensä Väestöennuste ' + str(year) + '"'
    def _parse_int(s):
        if s == ".":
            return None
        else:
            return int(s)
    result = []
    for line_idx, line in enumerate(all_lines[3:]):
        cells = line.strip().split(";")
        year = int(cells[0].strip('"'))
        assert 1900 <= year <= 2100
        assert cells[1] == '"Kuolleet"'
        deaths_actual = _parse_int(cells[2])
        output = {
            "year": year,
            "deaths_actual": deaths_actual,
        }
        for index, year in enumerate(years):
            deaths_forecast = _parse_int(cells[3+index])
            output["deaths_forecast_" + str(year)] = deaths_forecast
        result.append(output)
    return result

# Parse CSV file downloaded from:
# https://pxnet2.stat.fi/PXWeb/pxweb/fi/StatFin/StatFin__vrm__vamuu/statfin_vamuu_pxt_11ll.px/
# 
# Choices for obtaining CSV:
#   Tiedot = Kuolleet, Väkiluku
#   Kuukausi = Valitse kaikki
# 
# Click: Lataa taulukko > Lataa puolipiste-eroteltu csv-tiedosto (otsikollinen)
#
def parse_finland_deaths_and_population_by_month_csv(csv_file, trim_months_from_end):
    all_lines = list(csv_file.readlines())
    heading_line = all_lines[2].strip()
    heading_cells = heading_line.split(";")
    assert heading_cells[0] == '"Kuukausi"'
    assert heading_cells[1] == '"Kuolleet"'
    assert heading_cells[2] == '"Väkiluku"'
    result = []
    for line_idx, line in enumerate(all_lines[3:]):
        cells = line.strip().split(";")
        year_month_name = cells[0].strip('"').rstrip("*")
        year_s, month_s = year_month_name.split("M")
        year = int(year_s)
        month = int(month_s)
        assert 1900 <= year <= 2100
        assert 1 <= month <= 12
        deaths = int(cells[1])
        population = int(cells[2])
        output = {
            "year": year,
            "month": month,
            "deaths": deaths,
            "population": population,
            "deaths_per_100k": deaths / (population / 100000),
        }
        result.append(output)
    return result[:-trim_months_from_end]

# Parse CSV file downloaded from:
# https://pxnet2.stat.fi/PXWeb/pxweb/fi/StatFin/StatFin__vrm__kuol/statfin_kuol_pxt_12ah.px/
# 
# Choices for obtaining CSV:
#   Tiedot = Kuolleet
#   Vuosi = Valitse kaikki
#   -- Valitse luokitus -- = Kuukausi
#   Tapahtumakuukausi = Valitse kaikki
# 
# Click: Lataa taulukko > Lataa puolipiste-eroteltu csv-tiedosto (otsikollinen)
#
def parse_finland_deaths_by_month_csv(csv_file):
    all_lines = list(csv_file.readlines())
    heading_line = all_lines[2].strip()
    heading_cells = heading_line.split(";")
    assert heading_cells[0] == '"Vuosi"'
    assert heading_cells[1] == '"Kuukaudet yhteensä Kuolleet"'
    assert heading_cells[2] == '"Tammikuu Kuolleet"'
    assert heading_cells[-1] == '"Joulukuu Kuolleet"'
    result = []
    for line_idx, line in enumerate(all_lines[3:]):
        cells = line.strip().split(";")
        year = int(cells[0].strip('"'))
        assert 1900 <= year <= 2100
        deaths_total = int(cells[1])
        output = {
            "year": year,
            "deaths_total": deaths_total,
        }
        deaths_sum = 0
        for month in range(1, 13):
            month_deaths = int(cells[2+month-1])
            output[month] = month_deaths
            deaths_sum += month_deaths
        assert deaths_sum == deaths_total
        result.append(output)
    return result

# Parse CSV file downloaded from:
# https://pxnet2.stat.fi/PXWeb/pxweb/fi/StatFin/StatFin__vrm__vaerak/statfin_vaerak_pxt_11rb.px/table/tableViewLayout1/
# 
# Choices for obtaining CSV:
#   Tiedot = Väestö 31.12
#   Vuosi = Valitse kaikki
#   Sukupuoli = Valitse kaikki
# 
# Click: Lataa taulukko > Lataa puolipiste-eroteltu csv-tiedosto (otsikollinen)
#
def parse_finland_population_by_month_csv(csv_file):
    all_lines = list(csv_file.readlines())
    heading_line = all_lines[2].strip()
    heading_cells = heading_line.split(";")
    assert heading_cells[0] == '"Vuosi"'
    assert heading_cells[1] == '"Yhteensä Väestö 31.12."'
    result = []
    for line_idx, line in enumerate(all_lines[3:]):
        cells = line.strip().split(";")
        year = int(cells[0].strip('"'))
        assert 1900 <= year <= 2100
        population = int(cells[1])
        assert 200000 <= population <= 10000000
        result.append((year, population))
    return result

def calculate_moving_average(arr, w):
    results = []
    for i in range(len(arr)):
        lower_i = max(0, i-((w-w//2)-1))
        upper_i = min(len(arr)-1, i+w//2)
        window = arr[lower_i:upper_i+1]
        result = sum(window) / len(window)
        results.append(result)
    return numpy.array(results)

def test_calculate_moving_average():
    assert repr(list(calculate_moving_average([], 3))) == repr([])
    source2 = [1.0]
    result2 = list(calculate_moving_average(source2, 3))
    expected2 = [1.0]
    assert repr(result2) == repr(expected2), \
        "%s not equal to %s" % (repr(result2), repr(expected2))
    assert repr(list(calculate_moving_average([1.0, 3.0], 3))) == repr([2.0, 2.0])
    source4 = [90.0, 40.0, 95.0, 119.0, 102.0, 118.0, 33.0, 100.0, 4.0, 46.0, 109.0, 34.0, 25.0, 125.0, 120.0]
    result4 = list(calculate_moving_average(numpy.array(source4), 5))
    expected4 = [75.0, 86.0, 89.2, 94.8, 93.4, 94.4, 71.4, 60.2, 58.4, 58.6, 43.6, 67.8, 82.6, 76.0, 90.0]
    assert repr(result4) == repr(expected4), \
        "%s not equal to %s" % (repr(result4), repr(expected4))
test_calculate_moving_average()

def map_datetime_to_x(d):
    if isinstance(d, numpy.ndarray):
        return numpy.array([(d-T0_DATE).days for d in d])
    else:
        return (d - T0_DATE).days

def map_datetime64_to_x(d):
    return (d - T0_DATETIME64) / numpy.timedelta64(1, 'D')

def calculate_trend_extrapolation_slope(all_cause_mortality, years):
    acm_baseline_x_list = []
    acm_baseline_x_date_list = []
    acm_baseline_y_list = []
    start_date = BASELINE_CUTOFF_DATE - dateutil.relativedelta.relativedelta(years=years)
    for item_date, item_deaths in all_cause_mortality:
        if start_date <= item_date < BASELINE_CUTOFF_DATE:
            x = (item_date - T0_DATE).days
            acm_baseline_x_list.append(x)
            acm_baseline_x_date_list.append(item_date)
            acm_baseline_y_list.append(item_deaths)
    acm_baseline_x = numpy.array(acm_baseline_x_list)
    acm_baseline_x_date = numpy.array(acm_baseline_x_date_list)
    acm_baseline_y = numpy.array(acm_baseline_y_list)
    a, b = numpy.polyfit(acm_baseline_x, acm_baseline_y, 1)
    return a    

def calculate_acm_baseline_method1(all_cause_mortality, all_cause_mortality_estimate):
    acm_raw_x_list = []
    acm_raw_x_date_list = []
    acm_raw_y_list = []
    acm_baseline_x_list = []
    acm_baseline_x_date_list = []
    acm_baseline_y_list = []
    for item_date, item_deaths in all_cause_mortality:
        if item_date < BASELINE_START_DATE:
            continue
        x = (item_date - T0_DATE).days
        acm_raw_x_list.append(x)
        acm_raw_x_date_list.append(item_date)
        acm_raw_y_list.append(item_deaths)
        if item_date < BASELINE_CUTOFF_DATE:
            acm_baseline_x_list.append(x)
            acm_baseline_x_date_list.append(item_date)
            acm_baseline_y_list.append(item_deaths)
    acm_raw_x = numpy.array(acm_raw_x_list)
    acm_raw_x_date = numpy.array(acm_raw_x_date_list)
    acm_raw_y = numpy.array(acm_raw_y_list)
    acm_baseline_x = numpy.array(acm_baseline_x_list)
    acm_baseline_x_date = numpy.array(acm_baseline_x_date_list)
    acm_baseline_y = numpy.array(acm_baseline_y_list)

    acm_averaged_x = acm_raw_x
    acm_averaged_x_date = acm_raw_x_date
    acm_averaged_y = calculate_moving_average(acm_raw_y, 7)

    baseline_average_x_list = []
    baseline_average_x_date_list = []
    x_date = min(acm_raw_x_date) + BASELINE_INTERVAL / 2
    while x_date < BASELINE_CUTOFF_DATE:
        baseline_average_x_list.append((x_date - T0_DATE).days)
        baseline_average_x_date_list.append(x_date)
        x_date += BASELINE_INTERVAL
    baseline_average_x = numpy.array(baseline_average_x_list)
    baseline_average_x_date = numpy.array(baseline_average_x_date_list)
    baseline_average_y_list = []
    for baseline_x in baseline_average_x_date_list:
        window_start_date = baseline_x - BASELINE_INTERVAL / 2
        window_end_date = baseline_x + BASELINE_INTERVAL / 2
        average_sum = 0
        average_count = 0
        for acm_date, acm_y in zip(acm_raw_x_date, acm_raw_y):
            if acm_date >= window_start_date and acm_date < window_end_date:
                average_sum += acm_y
                average_count += 1
        baseline_y = average_sum / average_count
        baseline_average_y_list.append(baseline_y)
    baseline_average_y = numpy.array(baseline_average_y_list)
    
    if all_cause_mortality_estimate is not None:
        acm_estimate_x_list = [baseline_average_x[-1]]
        acm_estimate_x_date_list = [baseline_average_x_date[-1]]
        acm_estimate_y_list = [baseline_average_y[-1]]
        for x_date, year_mortality_estimate in all_cause_mortality_estimate:
            x = (x_date - T0_DATE).days
            acm_estimate_x_list.append(x)
            acm_estimate_x_date_list.append(x_date)
            acm_estimate_y_list.append(year_mortality_estimate/(datetime.date(x_date.year, 12, 31)-datetime.date(x_date.year, 1, 1)).days)
        acm_estimate_x = numpy.array(acm_estimate_x_list)
        acm_estimate_x_date = numpy.array(acm_estimate_x_date_list)
        acm_estimate_y = numpy.array(acm_estimate_y_list)
    else:
        acm_estimate_x = None
        acm_estimate_x_date = None
        acm_estimate_y = None
    if False:
        # spline trend
        spline_only_fn = scipy.interpolate.PchipInterpolator(baseline_average_x, baseline_average_y, axis=0, extrapolate=True)
        if all_cause_mortality_estimate is not None:
            def baseline_trend_fn(x):
                y = spline_only_fn(x)
                y[x > acm_estimate_x[0]]= acm_estimate_y[0] + (x[x>acm_estimate_x[0]] - acm_estimate_x[0]) * (acm_estimate_y[1] - acm_estimate_y[0]) / (acm_estimate_x[1] - acm_estimate_x[0])
                return y
        else:
            baseline_trend_fn = spline_only_fn
    else:
        # linear trend
        baseline_average_and_estimate_x = numpy.concatenate((baseline_average_x, acm_estimate_x[1:]))
        baseline_average_and_estimate_x_date = numpy.concatenate((baseline_average_x_date, acm_estimate_x_date[1:]))
        baseline_average_and_estimate_y = numpy.concatenate((baseline_average_y, acm_estimate_y[1:]))
        baseline_trend_fn = scipy.interpolate.interp1d(baseline_average_and_estimate_x, baseline_average_and_estimate_y, axis=0, fill_value="extrapolate")
        
    def days_to_rad(days):
        return 1 / DAYS_IN_YEAR_EXACT * 2 * math.pi * days
        
    def baseline_cosine_fn(x, t_offs, a):
        baseline = baseline_trend_fn(x)
        return baseline * (1 + a * numpy.cos(days_to_rad(x + t_offs)))

    baseline_cosine_params, _ = scipy.optimize.curve_fit(baseline_cosine_fn, acm_baseline_x, acm_baseline_y, p0=[-51, 0])
    print("Cosine time offset %.2f days (%.3f rad)" % (baseline_cosine_params[0], days_to_rad(baseline_cosine_params[0])))
    print("Cosine amplitude factor: %.4f" % (baseline_cosine_params[1],))
    def baseline_fn(x):
        return baseline_cosine_fn(x, *baseline_cosine_params)

    excess_mortality_x_list = []
    excess_mortality_x_date_list = []
    excess_mortality_y_list = []
    excess_mortality_baseline_part_sum = 0
    excess_mortality_baseline_part_count = 0
    baseline_part_deaths_sum = 0
    for item_date, item_deaths in all_cause_mortality:
        x = (item_date - T0_DATE).days
        baseline_deaths = baseline_fn(numpy.array([x]))[0]
        excess_deaths = item_deaths - baseline_deaths
        if item_date < BASELINE_CUTOFF_DATE:
            excess_mortality_baseline_part_sum += excess_deaths
            excess_mortality_baseline_part_count += 1
            baseline_part_deaths_sum += item_deaths
        excess_mortality_x_list.append(x)
        excess_mortality_x_date_list.append(item_date)
        excess_mortality_y_list.append(excess_deaths)
    excess_mortality_x = numpy.array(excess_mortality_x_list)
    excess_mortality_x_date = numpy.array(excess_mortality_x_date_list)
    excess_mortality_y = numpy.array(excess_mortality_y_list)
    if excess_mortality_baseline_part_count > 0:
        excess_mortality_baseline_part_avg = excess_mortality_baseline_part_sum / excess_mortality_baseline_part_count
    else:
        excess_mortality_baseline_part_avg = 0
    print("Sum of excess deaths: %.1f of %d total deaths" % (excess_mortality_baseline_part_sum, baseline_part_deaths_sum))

    output_dataseries("data_output/excess_mortality.csv", 
                      ["excess_mortality_x", "excess_mortality_x_date", "excess_mortality_y"],
                      excess_mortality_x, excess_mortality_x_date, excess_mortality_y)
    return (
        (acm_raw_x, acm_raw_x_date, acm_raw_y),
        (acm_averaged_x, acm_averaged_x_date, acm_averaged_y),
        (baseline_average_x, baseline_average_x_date, baseline_average_y),
        (acm_estimate_x, acm_estimate_x_date, acm_estimate_y),
        (baseline_trend_fn, baseline_fn),
        (excess_mortality_x, excess_mortality_x_date, excess_mortality_y)
    )

def calculate_acm_baseline_method2(all_cause_mortality, all_cause_mortality_estimate):
    acm_raw_x_list = []
    acm_raw_x_date_list = []
    acm_raw_y_list = []
    acm_baseline_x_list = []
    acm_baseline_x_date_list = []
    acm_baseline_y_list = []
    for item_date, item_deaths in all_cause_mortality:
        x = (item_date - T0_DATE).days
        acm_raw_x_list.append(x)
        acm_raw_x_date_list.append(item_date)
        acm_raw_y_list.append(item_deaths)
        if BASELINE_START_DATE <= item_date < BASELINE_CUTOFF_DATE:
            acm_baseline_x_list.append(x)
            acm_baseline_x_date_list.append(item_date)
            acm_baseline_y_list.append(item_deaths)
    acm_raw_x = numpy.array(acm_raw_x_list)
    acm_raw_x_date = numpy.array(acm_raw_x_date_list)
    acm_raw_y = numpy.array(acm_raw_y_list)
    acm_baseline_x = numpy.array(acm_baseline_x_list)
    acm_baseline_x_date = numpy.array(acm_baseline_x_date_list)
    acm_baseline_y = numpy.array(acm_baseline_y_list)

    acm_averaged_x = acm_raw_x
    acm_averaged_x_date = acm_raw_x_date
    acm_averaged_y = calculate_moving_average(acm_raw_y, 7)

    baseline_point_x_list = []
    baseline_point_x_date_list = []
    min_date = min(acm_baseline_x_date)
    max_date = max(acm_baseline_x_date)
    for x_date in (min_date, max_date):
        baseline_point_x_list.append((x_date - T0_DATE).days)
        baseline_point_x_date_list.append(x_date)
    #for year in range(min_date.year+1, max_date.year+10, 3):
    #    x_date = datetime.date(year, 1, 1) + datetime.timedelta(days=182)
    #    if min_date <= x_date <= max_date and x_date < BASELINE_CUTOFF_DATE:
    #        baseline_point_x_list.append((x_date - T0_DATE).days)
    #        baseline_point_x_date_list.append(x_date)
    baseline_point_x = numpy.array(baseline_point_x_list)
    baseline_point_x_date = numpy.array(baseline_point_x_date_list)
    baseline_average_y_list = []
    for baseline_x in baseline_point_x_date_list:
        window_size = datetime.timedelta(days=365*3)
        window_start_date = baseline_x - window_size / 2
        window_end_date = baseline_x + window_size / 2
        average_sum = 0
        average_count = 0
        for acm_date, acm_y in zip(acm_raw_x_date, acm_raw_y):
            if acm_date >= window_start_date and acm_date < window_end_date:
                average_sum += acm_y
                average_count += 1
        baseline_y = average_sum / average_count
        baseline_average_y_list.append(baseline_y)
    baseline_average_y = numpy.array(baseline_average_y_list)

    if False:
        print("Years\tTrend extrapolation slope")
        for i in (5, 6, 7, 8, 9, 10, 11):
            trend_extrapolation_slope = calculate_trend_extrapolation_slope(all_cause_mortality, years=i)
            print("%d\t%.5f" % (i, trend_extrapolation_slope))
        trend_extrapolation_slope = calculate_trend_extrapolation_slope(all_cause_mortality, years=10)
    else:
        trend_extrapolation_slope = None
    
    print("Year\t3 year average weekly dead")
    for x_date, deaths_average in zip(baseline_point_x_date, baseline_average_y):
        print("%s\t%.1f" % (x_date.year, deaths_average))
    if all_cause_mortality_estimate is not None:
        print("Year\tEstimated average weekly dead")
        acm_estimate_x_list = [baseline_point_x[-1]]
        acm_estimate_x_date_list = [baseline_point_x_date[-1]]
        acm_estimate_y_list = [baseline_average_y[-1]]
        for x_date, estimated_weekly_mortality in all_cause_mortality_estimate:
            x = (x_date - T0_DATE).days
            acm_estimate_x_list.append(x)
            acm_estimate_x_date_list.append(x_date)
            print("%s\t%.1f" % (x_date.year, estimated_weekly_mortality))
            acm_estimate_y_list.append(estimated_weekly_mortality)
        acm_estimate_x = numpy.array(acm_estimate_x_list)
        acm_estimate_x_date = numpy.array(acm_estimate_x_date_list)
        acm_estimate_y = numpy.array(acm_estimate_y_list)
    else:
        acm_estimate_x = None
        acm_estimate_x_date = None
        acm_estimate_y = None

    # linear interpolation between points found through curve fitting, where the point y-values are curve fitting variables
    #
        
    def get_linfit_fn(yvalues):
        linfit_x = baseline_point_x
        if acm_estimate_x is not None:
            linfit_x = numpy.concatenate((linfit_x, acm_estimate_x[1:]))
        linfit_y = numpy.array(yvalues)
        if acm_estimate_x is not None:
            linfit_y = numpy.concatenate((linfit_y, acm_estimate_y[1:]))
        if trend_extrapolation_slope is not None:
            delta_x = linfit_x[-1] - linfit_x[-2]
            trend_extrapolation_x = linfit_x[-1] + (linfit_x[-1] - linfit_x[-2])
            trend_extrapolation_y = linfit_y[-1] + delta_x * trend_extrapolation_slope
            linfit_x = numpy.concatenate((linfit_x, numpy.array([trend_extrapolation_x])))
            linfit_y = numpy.concatenate((linfit_y, numpy.array([trend_extrapolation_y])))
        return scipy.interpolate.interp1d(linfit_x, linfit_y, axis=0, fill_value="extrapolate")
        
    def days_to_rad(days):
        return 1 / DAYS_IN_YEAR_EXACT * 2 * math.pi * days
        
    def baseline_optimization_fn(x, t_offs, a, *yvalues):
        assert len(yvalues) == len(baseline_point_x), "Invalid lengths: %s != %s" % (len(yvalues), len(baseline_point_x))
        linfit_fn = get_linfit_fn(yvalues)
        trend_value = linfit_fn(x)
        return trend_value * (1 + a * numpy.cos(days_to_rad(x + t_offs)))

    # Reason for this hack is that curve_fit inspects the optimization function and requires 
    # positional parameters, i.e. variable *args is not possible. This array allows for
    # variable amount of piecewise linear points
    fn_wrappers = [lambda x, t_offs, a, p0: baseline_optimization_fn(x, t_offs, a, p0),
                   lambda x, t_offs, a, p0, p1: baseline_optimization_fn(x, t_offs, a, p0, p1),
                   lambda x, t_offs, a, p0, p1, p2: baseline_optimization_fn(x, t_offs, a, p0, p1, p2),
                   lambda x, t_offs, a, p0, p1, p2, p3: baseline_optimization_fn(x, t_offs, a, p0, p1, p2, p3),
                   lambda x, t_offs, a, p0, p1, p2, p3, p4: baseline_optimization_fn(x, t_offs, a, p0, p1, p2, p3, p4),
                   lambda x, t_offs, a, p0, p1, p2, p3, p4, p5: baseline_optimization_fn(x, t_offs, a, p0, p1, p2, p3, p4, p5),
                   lambda x, t_offs, a, p0, p1, p2, p3, p4, p5, p6: baseline_optimization_fn(x, t_offs, a, p0, p1, p2, p3, p4, p5, p6),
                   lambda x, t_offs, a, p0, p1, p2, p3, p4, p5, p6, p7: baseline_optimization_fn(x, t_offs, a, p0, p1, p2, p3, p4, p5, p6, p7),
                   lambda x, t_offs, a, p0, p1, p2, p3, p4, p5, p6, p7, p8: baseline_optimization_fn(x, t_offs, a, p0, p1, p2, p3, p4, p5, p6, p7, p8),
                   lambda x, t_offs, a, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9: baseline_optimization_fn(x, t_offs, a, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9),
                   lambda x, t_offs, a, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10: baseline_optimization_fn(x, t_offs, a, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10),
                   lambda x, t_offs, a, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11: baseline_optimization_fn(x, t_offs, a, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11),
                   lambda x, t_offs, a, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12: baseline_optimization_fn(x, t_offs, a, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12),
                   lambda x, t_offs, a, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13: baseline_optimization_fn(x, t_offs, a, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13),
                   lambda x, t_offs, a, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14: baseline_optimization_fn(x, t_offs, a, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14),
                   lambda x, t_offs, a, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15: baseline_optimization_fn(x, t_offs, a, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15),
                   lambda x, t_offs, a, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16: baseline_optimization_fn(x, t_offs, a, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16),
                   lambda x, t_offs, a, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17: baseline_optimization_fn(x, t_offs, a, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17),
                   ]
    optimization_values, _ = scipy.optimize.curve_fit(fn_wrappers[len(baseline_point_x)-1], acm_baseline_x, acm_baseline_y)
    baseline_trend_fn = get_linfit_fn(optimization_values[2:])
    print("Cosine time offset %.2f days (%.3f rad)" % (optimization_values[0], days_to_rad(optimization_values[0])))
    print("Cosine amplitude factor: %.4f" % (optimization_values[1],))
    
    def baseline_fn(x):
        return baseline_optimization_fn(x, *optimization_values)

    excess_mortality_x_list = []
    excess_mortality_x_date_list = []
    excess_mortality_y_list = []
    excess_mortality_baseline_part_sum = 0
    excess_mortality_baseline_part_count = 0
    baseline_part_deaths_sum = 0
    for item_date, item_deaths in all_cause_mortality:
        x = (item_date - T0_DATE).days
        baseline_deaths = baseline_fn(numpy.array([x]))[0]
        excess_deaths = item_deaths - baseline_deaths
        if item_date < BASELINE_CUTOFF_DATE:
            excess_mortality_baseline_part_sum += excess_deaths
            excess_mortality_baseline_part_count += 1
            baseline_part_deaths_sum += item_deaths
        excess_mortality_x_list.append(x)
        excess_mortality_x_date_list.append(item_date)
        excess_mortality_y_list.append(excess_deaths)
    excess_mortality_x = numpy.array(excess_mortality_x_list)
    excess_mortality_x_date = numpy.array(excess_mortality_x_date_list)
    excess_mortality_y = numpy.array(excess_mortality_y_list)
    if excess_mortality_baseline_part_count > 0:
        excess_mortality_baseline_part_avg = excess_mortality_baseline_part_sum / excess_mortality_baseline_part_count
    else:
        excess_mortality_baseline_part_avg = 0
    print("Sum of excess deaths: %.1f of %d total deaths" % (excess_mortality_baseline_part_sum, baseline_part_deaths_sum))

    excess_mortality_week = []
    for x_date in excess_mortality_x_date:
        year, week, _ = x_date.isocalendar()
        excess_mortality_week.append("%dW%02d" % (year, week))
    output_dataseries("data_output/excess_mortality.csv", 
                      ["excess_mortality_x", "excess_mortality_x_date", "excess_mortality_week", "excess_mortality_y"],
                      excess_mortality_x, excess_mortality_x_date, excess_mortality_week, excess_mortality_y)
    return (
        (acm_raw_x, acm_raw_x_date, acm_raw_y),
        (acm_averaged_x, acm_averaged_x_date, acm_averaged_y),
        (baseline_point_x, baseline_point_x_date, baseline_trend_fn(baseline_point_x)),
        (acm_estimate_x, acm_estimate_x_date, acm_estimate_y),
        (baseline_trend_fn, baseline_fn),
        (excess_mortality_x, excess_mortality_x_date, excess_mortality_y)
    )

def calculate_life_expectancy_fn(country_life_expectancy):
    life_expectancy_x = []
    life_expectancy_x_date = []
    life_expectancy_both_y = []
    life_expectancy_male_y = []
    life_expectancy_female_y = []
    for year, both_life_expectancy, male_life_expectancy, female_life_expectancy in country_life_expectancy:
        x_date = datetime.date(year, 1, 1)
        x = (x_date - T0_DATE).days
        life_expectancy_x.append(x)
        life_expectancy_x_date.append(x_date)
        life_expectancy_both_y.append(both_life_expectancy)
        life_expectancy_male_y.append(male_life_expectancy)
        life_expectancy_female_y.append(female_life_expectancy)
    both_life_expectancy_fn = scipy.interpolate.interp1d(life_expectancy_x, life_expectancy_both_y, axis=0)
    male_life_expectancy_fn = scipy.interpolate.interp1d(life_expectancy_x, life_expectancy_male_y, axis=0)
    female_life_expectancy_fn = scipy.interpolate.interp1d(life_expectancy_x, life_expectancy_female_y, axis=0)
    return both_life_expectancy_fn, male_life_expectancy_fn, female_life_expectancy_fn


def combine_deaths_by_month(finland_deaths_by_month_since_1945, finland_deaths_and_population_by_month, min_year):
    deaths_by_year_and_month = {}
    for year_item in finland_deaths_by_month_since_1945:
        year = year_item["year"]
        for month in range(1, 13):
            key = (year, month)
            deaths_by_year_and_month[key] = year_item[month]
    for month_item in finland_deaths_and_population_by_month:
        year = month_item["year"]
        month = month_item["month"]
        key = (year, month)
        if key in deaths_by_year_and_month:
            assert month_item["deaths"] == deaths_by_year_and_month[key]
        else:
            deaths_by_year_and_month[key] = month_item["deaths"]
    all_years = set((x[0] for x in deaths_by_year_and_month.keys()))
    if min_year is None:
        min_year = min(all_years)
    max_year = max(all_years)
    result = []
    for year in range(min_year, max_year+1):
        if year < min_year:
            continue
        for month in range(1, 13):
            date = datetime.date(year, month, 1)
            key = (year, month)
            deaths = deaths_by_year_and_month.get(key)
            if year < datetime.date.today().year:
                assert deaths is not None, "Year %d month %d not in result" % (year, month)
            if deaths is not None:
                result.append({
                    "year": year,
                    "month": month,
                    "deaths": deaths,
                })
    return result

def split_acm_by_cutoff_date(acm_averaged_x, acm_averaged_x_date, acm_averaged_y):
    baseline_part_x_list = []
    baseline_part_x_date_list = []
    baseline_part_y_list = []
    nonbaseline_part_x_list = []
    nonbaseline_part_x_date_list = []
    nonbaseline_part_y_list = []
    for x, d, y in zip(acm_averaged_x, acm_averaged_x_date, acm_averaged_y):
        if d < BASELINE_CUTOFF_DATE:
            baseline_part_x_list.append(x)
            baseline_part_x_date_list.append(d)
            baseline_part_y_list.append(y)
        else:
            nonbaseline_part_x_list.append(x)
            nonbaseline_part_x_date_list.append(d)
            nonbaseline_part_y_list.append(y)
    baseline_part_x = numpy.array(baseline_part_x_list)
    baseline_part_x_date = numpy.array(baseline_part_x_date_list)
    baseline_part_y = numpy.array(baseline_part_y_list)
    nonbaseline_part_x = numpy.array(nonbaseline_part_x_list)
    nonbaseline_part_x_date = numpy.array(nonbaseline_part_x_date_list)
    nonbaseline_part_y = numpy.array(nonbaseline_part_y_list)
    return (baseline_part_x, baseline_part_x_date, baseline_part_y,
            nonbaseline_part_x, nonbaseline_part_x_date, nonbaseline_part_y)

def output_dataseries(output_filename, headings, *data_columns):
    assert len(headings) == len(data_columns)
    with open(output_filename, "wt", encoding="utf-8", newline='\r\n') as outf:
        def write_line(cols):
            outf.write(','.join(['"' + str(col).replace('"', '""') + '"' for col in cols]) + "\n")
        write_line(headings)
        for tupl in zip(*data_columns):
            write_line(tupl)
            
def get_model_yearly_mortality(baseline_fn, year):
    a = (datetime.date(year, 1, 1) - T0_DATE).days
    b = (datetime.date(year+1, 1, 1) - T0_DATE).days
    y, err = scipy.integrate.quad(baseline_fn, a, b)
    return round(y/7)

def plot_population_forecast_vs_model(population_forecast, baseline_fn):
    plot_x_list = []
    plot_model_list = []
    plot_actual_list = []
    plot_fc2007_list = []
    plot_fc2009_list = []
    plot_fc2012_list = []
    plot_fc2015_list = []
    plot_fc2018_list = []
    plot_fc2019_list = []
    plot_fc2021_list = []
    plot_fc2007_inactive_list = []
    plot_fc2009_inactive_list = []
    plot_fc2012_inactive_list = []
    plot_fc2015_inactive_list = []
    plot_fc2018_inactive_list = []
    plot_fc2019_inactive_list = []
    plot_fc2021_inactive_list = []
    forecast_years = (2021, 2019, 2018, 2015, 2012, 2009, 2007)
    for year_values in sorted(population_forecast, key=lambda x: x["year"]):
        year = year_values["year"]
        plot_x_list.append(year)
        model_deaths = round(get_model_yearly_mortality(baseline_fn, year))
        plot_model_list.append(model_deaths)
        if year == 2021:
            deaths_actual = 57343
        else:
            deaths_actual = year_values.get("deaths_actual")
        plot_actual_list.append(deaths_actual)

        has_last_value = False
        prev_target_year = None
        for (target_year, 
             target_list, 
             target_inactive_list, 
             source_value) in ((2021, plot_fc2021_list, plot_fc2021_inactive_list, "deaths_forecast_2021"),
                               (2019, plot_fc2019_list, plot_fc2019_inactive_list, "deaths_forecast_2019"),
                               (2018, plot_fc2018_list, plot_fc2018_inactive_list, "deaths_forecast_2018"),
                               (2015, plot_fc2015_list, plot_fc2015_inactive_list, "deaths_forecast_2015"),
                               (2012, plot_fc2012_list, plot_fc2012_inactive_list, "deaths_forecast_2012"),
                               (2009, plot_fc2009_list, plot_fc2009_inactive_list, "deaths_forecast_2009"),
                               (2007, plot_fc2007_list, plot_fc2007_inactive_list, "deaths_forecast_2007")):
            value = year_values.get(source_value)
            if value is not None:
                if (not has_last_value or year == prev_target_year):
                    target_list.append(value)
                    if has_last_value:
                        target_inactive_list.append(value)
                    else:
                        target_inactive_list.append(None)
                    has_last_value = True
                    prev_target_year = target_year
                else:
                    target_list.append(None)
                    target_inactive_list.append(value)
            else:
                target_list.append(None)
                target_inactive_list.append(None)
    plot_x = numpy.array(plot_x_list)
    plot_actual = numpy.array(plot_actual_list)
    plot_model = numpy.array(plot_model_list)
    plot_fc2007 = numpy.array(plot_fc2007_list)
    plot_fc2009 = numpy.array(plot_fc2009_list)
    plot_fc2012 = numpy.array(plot_fc2012_list)
    plot_fc2015 = numpy.array(plot_fc2015_list)
    plot_fc2018 = numpy.array(plot_fc2018_list)
    plot_fc2019 = numpy.array(plot_fc2019_list)
    plot_fc2021 = numpy.array(plot_fc2021_list)
    plot_fc2007_inactive = numpy.array(plot_fc2007_inactive_list)
    plot_fc2009_inactive = numpy.array(plot_fc2009_inactive_list)
    plot_fc2012_inactive = numpy.array(plot_fc2012_inactive_list)
    plot_fc2015_inactive = numpy.array(plot_fc2015_inactive_list)
    plot_fc2018_inactive = numpy.array(plot_fc2018_inactive_list)
    plot_fc2019_inactive = numpy.array(plot_fc2019_inactive_list)
    plot_fc2021_inactive = numpy.array(plot_fc2021_inactive_list)

    fig, ax = plt.subplots(1, 1, figsize=(COLUMN_WIDTH_IN, 1.5))
    plt.ylim(46000, 62000)
    plt.xlim(2010, 2030)
    active_lw = 0.7
    inactive_lw = 0.25
    line_actual, = ax.plot(plot_x, plot_actual, label="Kuolin-\ntilasto$^*$", color="C0", marker='o', markersize=1, linewidth=DEFAULT_LINEWIDTH, zorder=1)
    line_model, = ax.plot(plot_x, plot_model, label="Malli", color="C3", linewidth=DEFAULT_LINEWIDTH, zorder=2)
    #line_fc2007, = ax.plot(plot_x, plot_fc2007, label="V-e 2007", color="C6", linewidth=active_lw)
    #line_fc2009, = ax.plot(plot_x, plot_fc2009, label="V-e 2009", color="C7", linewidth=active_lw)
    line_fc2012, = ax.plot(plot_x, plot_fc2012, label="V-e 2012", color="C5", linewidth=active_lw)
    ax.plot(plot_x, plot_fc2012_inactive, label=None, color="C5", linewidth=inactive_lw, linestyle="dashed")
    line_fc2015, = ax.plot(plot_x, plot_fc2015, label="V-e 2015", color="C2", linewidth=active_lw)
    ax.plot(plot_x, plot_fc2015_inactive, label=None, color="C2", linewidth=inactive_lw, linestyle="dashed")
    line_fc2018, = ax.plot(plot_x, plot_fc2018, label="V-e 2018", color="C4", linewidth=active_lw)
    ax.plot(plot_x, plot_fc2018_inactive, label=None, color="C4", linewidth=inactive_lw, linestyle="dashed")
    line_fc2019, = ax.plot(plot_x, plot_fc2019, label="V-e 2019", color="C8", linewidth=active_lw)
    ax.plot(plot_x, plot_fc2019_inactive, label=None, color="C8", linewidth=inactive_lw, linestyle="dashed")
    line_fc2021, = ax.plot(plot_x, plot_fc2021, label="V-e 2021", color="C1", linewidth=active_lw)
    ax.plot(plot_x, plot_fc2021_inactive, label=None, color="C1", linewidth=inactive_lw, linestyle="dashed")
    plt.grid(True)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(2000))
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
    ax.tick_params(axis="x", pad=0.5, labelsize=7, labelrotation=70)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment('center')
        label.set_verticalalignment('top')
    ax.legend(handles=[line_actual,
                       line_model,
                       line_fc2021,
                       line_fc2019,
                       line_fc2018,
                       line_fc2015,
                       line_fc2012,
                       #line_fc2009,
                       #line_fc2007,
                        ], 
               loc="upper left", ncol=1, bbox_to_anchor=(1.03,1), fontsize=7,
               borderpad=0.3, handletextpad=0.6, labelspacing=0.4, frameon=True, fancybox=False, 
               shadow=False, facecolor="white", framealpha=1.0, borderaxespad=0)
    fig.text(0.995, 0.22, "*) 2021: ennakkotieto", horizontalalignment="right", fontsize=6, transform=fig.transFigure)
    fig.subplots_adjust(left=0.095, right=0.76, top=0.97, bottom=0.15)
    plot_model_cutoff(fig, ax, xpos=2019.5, ypos=0.18)
    save_fig(fig, "figures/population_forecast_vs_model")
    #plt.show(block=True)
    plt.close(fig)

def plot_monthly_deaths_per_100k(deaths_and_population_by_month, covid_data):
    months_x = numpy.array(range(1, 13))
    by_month = dict(((x, []) for x in months_x))
    by_month_2021 = dict()
    by_month_2022 = dict()
    for month_item in deaths_and_population_by_month:
        year = month_item["year"]
        month = month_item["month"]
        if year < 2021:
            by_month[month].append(month_item)
        elif year == 2021:
            by_month_2021[month] = month_item
        elif year == 2022:
            by_month_2022[month] = month_item
    min_y = []
    max_y = []
    min80pc_y = []
    max80pc_y = []
    line2021_y = []
    line2022_y = []
    for month in months_x:
        month_items = by_month[month]
        month_deaths = list(map(lambda x: x["deaths_per_100k"], month_items))
        month_deaths.sort()
        min_y.append(month_deaths[0])
        max_y.append(month_deaths[-1])
        idx10pc = round(0.1*len(month_deaths))
        idx90pc = idx10pc + round(0.8*len(month_deaths))
        min80pc_y.append(month_deaths[idx10pc])
        max80pc_y.append(month_deaths[idx90pc])
        item2021 = by_month_2021.get(month)
        line2021_y.append(item2021["deaths_per_100k"] if item2021 is not None else None)
        item2022 = by_month_2022.get(month)
        line2022_y.append(item2022["deaths_per_100k"] if item2022 is not None else None)
    line_covid_2021_y = []
    cur_month = None
    cur_month_sum = None
    for x_date, covid_deaths, covid_cases, covid_tests in covid_data:
        if x_date.year != 2021:
            continue
        if cur_month is None or x_date.month != cur_month:
            if cur_month_sum is not None:
                population = by_month_2021[cur_month]["population"]
                line_covid_2021_y.append(cur_month_sum / (population / 100000))
            cur_month = x_date.month
            cur_month_sum = 0
        cur_month_sum += covid_deaths
    assert cur_month_sum is not None
    assert cur_month == 12
    population = by_month_2021[cur_month]["population"]
    line_covid_2021_y.append(cur_month_sum / (population / 100000))
    assert len(line_covid_2021_y) == 12
    fig, ax = plt.subplots(1, 1, figsize=(COLUMN_WIDTH_IN, 1.5))
    ax.set_ylim(0, 120)
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
    minmax_fill = ax.fill_between(months_x, max_y, min_y, label="Min-max 1990-2020", 
                                  facecolor="C0", linewidth=0.05, interpolate=True,
                                  alpha=0.4, edgecolor=None)
    minmax80pc_fill = ax.fill_between(months_x, max80pc_y, min80pc_y, label="80%", 
                                      facecolor="C0", linewidth=0.05, interpolate=True,
                                      alpha=1, edgecolor=None)
    line_2021, = ax.plot(months_x, line2021_y, label="2021", color="C1", marker='D', markersize=2, linewidth=DEFAULT_LINEWIDTH, zorder=10)
    #line_2022, = ax.plot(months_x, line2022_y, label="2022", color="C4", marker='*', markersize=2, linewidth=DEFAULT_LINEWIDTH, zorder=10)
    line_covid_2021, = ax.plot(months_x, line_covid_2021_y, label="2021 korona", color="C1", marker='o', markersize=1, linewidth=DEFAULT_LINEWIDTH, zorder=10)
    ax.legend(handles=[minmax_fill, minmax80pc_fill, line_2021, line_covid_2021],
              loc="upper left", bbox_to_anchor=(0, 1.15), ncol=4, fontsize=7,
              borderpad=0.3, handletextpad=0.25, columnspacing=1, frameon=True, fancybox=False, 
              shadow=False, facecolor="white", framealpha=1.0, borderaxespad=0.05)
    ax_ylabel_transform = matplotlib.transforms.blended_transform_factory(fig.transFigure, ax.transAxes)
    ax.text(0.02, 0.5, "Kuolleet per 100 000", transform=ax_ylabel_transform, fontsize=7, rotation=90,
            horizontalalignment="center", verticalalignment="center")
    ax.text(0.005, -0.035, "Kuukausi", transform=ax_ylabel_transform, fontsize=7,
            horizontalalignment="left", verticalalignment="top")
    fig.subplots_adjust(left=0.095, right=0.99, top=0.88, bottom=0.09)
    save_fig(fig, "figures/monthly_deaths_per_100k")
    #plt.show(block=True)
    plt.close(fig)
    
def plot_weekly_deaths_per_age_per_1M(population_by_year_and_age, acm_by_category):
    age_buckets = [
        {
            "name": "0-19 -vuotiaat",
            "age_categories": ["0-4", "5-9", "10-14", "15-19",],
            "age_start": 0,
            "age_end": 19,
        },
        {
            "name": "20-49 -vuotiaat",
            "age_categories": ["20-24", "25-29", "30-34", "35-39", "40-44", "45-49"],
            "age_start": 20,
            "age_end": 49,
        },
        {
            "name": "50-59 -vuotiaat",
            "age_categories": ["50-54", "55-59",],
            "age_start": 50,
            "age_end": 59,
        },
        {
            "name": "60-69 -vuotiaat",
            "age_categories": ["60-64", "65-69",],
            "age_start": 60,
            "age_end": 69,
        },
        {
            "name": "70-79 -vuotiaat",
            "age_categories": ["70-74", "75-79",],
            "age_start": 70,
            "age_end": 79,
        },
        {
            "name": "Yli 80-vuotiaat",
            "age_categories": ["80-84", "85-89", "90-"],
            "age_start": 80,
            "age_end": None,
        },
    ]
    for age_bucket in age_buckets:
        bucket_deaths_timeseries = []
        for age_category_index, age_category in enumerate(age_bucket["age_categories"]):
            for timeseries_index, (x_date, deaths) in enumerate(acm_by_category[age_category]):
                if age_category_index == 0:
                    bucket_deaths_timeseries.append([x_date, deaths])
                else:
                    item = bucket_deaths_timeseries[timeseries_index]
                    assert x_date == item[0]
                    item[1] += deaths
        year_series = {}
        for x_date, deaths in bucket_deaths_timeseries:
            year, week, _ = x_date.isocalendar()
            if year < 2000:
                continue
            population = 0
            if year not in population_by_year_and_age:
                population_year_key = max(population_by_year_and_age.keys())
            else:
                population_year_key = year
            for age_key, age_item in population_by_year_and_age[population_year_key].items():
                if isinstance(age_key, int):
                    if age_bucket["age_start"] <= age_key and (age_bucket["age_end"] is None or age_key <= age_bucket["age_end"]):
                        population += age_item[0]
            mortality = deaths / (population / 1000000.0)
            if year not in year_series:
                year_series[year] = {
                    "x": [],
                    "y": [],
                }
            assert week not in year_series[year]["x"]
            assert 1 <= week <= 53
            year_series[year]["x"].append(week)
            year_series[year]["y"].append(mortality)
        age_bucket["year_series"] = year_series

    xticks = [x for x in range(1, 54, 2)]
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(PAPER_WIDTH_IN, 6))
    all_axes = (ax1, ax2, ax3, ax4, ax5, ax6)
    for ax, age_bucket in zip(all_axes, age_buckets):
        ax.set_xticks(xticks)
        for year in range(2000, 2010):
            ax.plot(age_bucket["year_series"][year]["x"], age_bucket["year_series"][year]["y"], color="#dae2ef", linewidth=0.3, zorder=3)
        for year in range(2010, 2020):
            ax.plot(age_bucket["year_series"][year]["x"], age_bucket["year_series"][year]["y"], color="#8ba8c8", linewidth=0.3, zorder=3)
        ax.plot(age_bucket["year_series"][2020]["x"], age_bucket["year_series"][2020]["y"], color="#2165ac", linewidth=0.9, zorder=3)
        ax.plot(age_bucket["year_series"][2021]["x"], age_bucket["year_series"][2021]["y"], color="#b1182c", linewidth=0.9, zorder=3)
        ax.set_ylim(bottom=0, top=None, auto=True)
        ax.set_xlim(1, 53)
        ax.tick_params(labelsize=4)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment('center')
        ax.text(0.00, 1.01, age_bucket["name"], fontsize=9, horizontalalignment="left", verticalalignment="bottom", transform=ax.transAxes)
        ax.text(-0.085, 0.5, "Viikoittain kuolleet per miljoona", transform=ax.transAxes, fontsize=6, rotation=90,
                horizontalalignment="center", verticalalignment="center")
        ax.text(-0.03, -0.02, "Viikko", fontsize=5, horizontalalignment="right", verticalalignment="top", transform=ax.transAxes)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.97, bottom=0.05)
    save_fig(fig, "figures/weekly_deaths_per_age_per_1M_THL")

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(PAPER_WIDTH_IN, 6))
    all_axes = (ax1, ax2, ax3, ax4, ax5, ax6)
    colors = plt.cm.gist_rainbow(numpy.linspace(0,1,22))
    def _year_to_color(year):
        return colors[year-2000]
    for ax, age_bucket in zip(all_axes, age_buckets):
        ax.set_xticks(xticks)
        for year in range(2000, 2022):
            age_bucket["year_series"][year]["y_avg"] = calculate_moving_average(age_bucket["year_series"][year]["y"], 10)
        for year in range(2000, 2010):
            ax.plot(age_bucket["year_series"][year]["x"], age_bucket["year_series"][year]["y_avg"], 
                    c=_year_to_color(year), linewidth=0.5, zorder=3)
        for year in range(2010, 2020):
            ax.plot(age_bucket["year_series"][year]["x"], age_bucket["year_series"][year]["y_avg"], 
                    c=_year_to_color(year), linewidth=0.5, zorder=3)
        ax.plot(age_bucket["year_series"][2020]["x"], age_bucket["year_series"][2020]["y_avg"], 
                c=_year_to_color(2020), linewidth=1.5, zorder=4)
        ax.plot(age_bucket["year_series"][2021]["x"], age_bucket["year_series"][2021]["y_avg"], 
                c="#ff4444", linewidth=1.5, zorder=4)
        #ax.set_ylim(bottom=0, top=None, auto=True)
        ax.set_xlim(1, 53)
        ax.tick_params(labelsize=4)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment('center')
        ax.text(0.00, 1.01, age_bucket["name"], fontsize=9, horizontalalignment="left", verticalalignment="bottom", transform=ax.transAxes)
        ax.text(-0.085, 0.5, "Viikoittain kuolleet per miljoona", transform=ax.transAxes, fontsize=6, rotation=90,
                horizontalalignment="center", verticalalignment="center")
        ax.text(-0.03, -0.02, "Viikko", fontsize=5, horizontalalignment="right", verticalalignment="top", transform=ax.transAxes)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.97, bottom=0.05)
    save_fig(fig, "figures/weekly_deaths_per_age_per_1M_improved")
    #plt.show(block=True)
    plt.close(fig)

def plot_raw_acm(acm_raw_x, acm_raw_x_date, acm_raw_y, auto_limits):
    fig, ax = plt.subplots(1, 1, figsize=(PAPER_WIDTH_IN, 1.2))
    #ax.plot(deaths_x, deaths_y, color="red", linewidth=DEFAULT_LINEWIDTH, linestyle="-")
    #plt.scatter(deaths_x, deaths_y, color="red", s=0.5)
    #plt.xlim(min(acm_raw_x_date), max(acm_raw_x_date))
    ax.set_xlim(min(acm_raw_x_date), max(acm_raw_x_date))
    if not auto_limits:
        ax.set_ylim(800, 1400)
    ax.plot(acm_raw_x_date, acm_raw_y, color="C0", linewidth=0.2, linestyle="solid", zorder=1, alpha=0.5)
    acm_1yr_y = calculate_moving_average(acm_raw_y, 52)
    acm_2yr_y = calculate_moving_average(acm_raw_y, 52*2)
    acm_3yr_y = calculate_moving_average(acm_raw_y, 52*3)
    #ax.plot(acm_raw_x_date, acm_1yr_y, color="C1", linewidth=0.5, linestyle="solid", zorder=10)
    #ax.plot(acm_raw_x_date, acm_2yr_y, color="C1", linewidth=0.5, linestyle="solid", zorder=11)
    ax.plot(acm_raw_x_date, acm_3yr_y, color="C1", linewidth=0.5, linestyle="solid", zorder=12)
    fig.subplots_adjust(left=0.04, right=0.995, top=0.965, bottom=0.105)
    save_fig(fig, "figures/acm_raw")
    #plt.show(block=True)
    plt.close(fig)

BASELINE_FITTING_LINE_PROPS = {
    "color": "C0", 
    "linewidth": DEFAULT_LINEWIDTH, 
    "linestyle": "solid", 
}

BASELINE_NONFITTING_LINE_PROPS = {
    "color": "C0",                         # black
    "linewidth": DEFAULT_LINEWIDTH, 
    "linestyle": "solid",                  # dotted
}

CUTOFF_ARROW_LENGTH = datetime.timedelta(days=3*365)

def plot_model_cutoff(fig, ax, xpos=BASELINE_CUTOFF_DATE, ypos=0.14):
    # https://matplotlib.org/stable/gallery/pyplots/annotate_transform.html#sphx-glr-gallery-pyplots-annotate-transform-py
    def _optional_dateconvert(d):
        if isinstance(d, datetime.date):
            return matplotlib.dates.date2num(d)
        else:
            return d
    xlim_min, xlim_max = ax.get_xlim()
    xlim_min, xlim_max = _optional_dateconvert(xlim_min), _optional_dateconvert(xlim_max)
    xpos_frac = (_optional_dateconvert(xpos) - xlim_min) / (xlim_max - xlim_min)
    #ax.axvline(xpos_frac, linewidth=0.8, color="black", linestyle = "dashed", alpha=1, transform=ax.transAxes)
    vline = matplotlib.lines.Line2D([xpos_frac, xpos_frac], [1, 0], linewidth=0.6, color="black", 
                                    linestyle = "dashed", alpha=1, clip_on=False, transform=ax.transAxes, zorder=9)
    ax.add_line(vline)
    arrowprops = {
        "connectionstyle": "arc3", 
        "arrowstyle": "<-",
        "edgecolor": "black", 
        "facecolor": "black",
        "shrinkA": 0.15,
        "shrinkB": 0.01,
    }
    xoffset = 16
    bbox = {
        "boxstyle": "round",
        "facecolor": "white",
        "linewidth": 0.25,
        "edgecolor": "black",
        "pad": 0.1,
    }
    ax.annotate("Mallin muodostus", (xpos_frac, ypos), (-xoffset, 0), verticalalignment="center", horizontalalignment="right", 
                clip_on=False, arrowprops=arrowprops, fontsize=6, 
                xycoords="axes fraction", textcoords="offset points", zorder=9, color="black", bbox=bbox)
    ax.annotate("Vertailu", (xpos_frac, ypos), (xoffset, 0), verticalalignment="center", horizontalalignment="left", 
                clip_on=False, arrowprops=arrowprops, fontsize=6, 
                xycoords="axes fraction", textcoords="offset points", zorder=9, color="black", bbox=bbox)
    
def plot_acm_baseline_trend(acm_raw_x, acm_raw_x_date, acm_raw_y,
                             acm_averaged_x, acm_averaged_x_date, acm_averaged_y,
                             baseline_average_x, baseline_average_x_date, baseline_average_y,
                             acm_estimate_x, acm_estimate_x_date, acm_estimate_y,
                             baseline_trend_fn, auto_limits):
    fig, ax = plt.subplots(1, 1, figsize=(PAPER_WIDTH_IN, BASELINE_PLOT_HEIGHT))
    #ax.plot(deaths_x, deaths_y, color="red", linewidth=DEFAULT_LINEWIDTH, linestyle="-")
    #ax.scatter(deaths_x, deaths_y, color="red", s=0.5)
    #ax.set_xlim(min(acm_raw_x_date), max(acm_raw_x_date))
    ax.set_xlim(BASELINE_PLOT_START_DATE, BASELINE_PLOT_END_DATE)
    if not auto_limits:
        plt.ylim(*ACM_CUSTOM_YLIM)
    (
        baseline_part_x, baseline_part_x_date, baseline_part_y,
        nonbaseline_part_x, nonbaseline_part_x_date, nonbaseline_part_y,
    ) = split_acm_by_cutoff_date(acm_averaged_x, acm_averaged_x_date, acm_averaged_y)
    ax.plot(baseline_part_x_date, baseline_part_y, zorder=1, **BASELINE_FITTING_LINE_PROPS)
    ax.plot(nonbaseline_part_x_date, nonbaseline_part_y, zorder=1, **BASELINE_NONFITTING_LINE_PROPS)
    #ax.plot(baseline_average_x_date, baseline_average_y, color="olivedrab", linewidth=1, linestyle="-", zorder=2)

    baseline_range = pandas.date_range(datetime.date(1980, 1, 3), datetime.date(2030, 1, 1), periods=3000)
    #baseline_range = pandas.date_range(min(acm_raw_x), max(acm_raw_x), periods=200)
    baseline_trend_x_date = numpy.array(pandas.to_datetime(baseline_range))
    baseline_trend_x = map_datetime64_to_x(baseline_trend_x_date)
    baseline_trend_y = baseline_trend_fn(baseline_trend_x)
    ax.plot(baseline_trend_x_date, baseline_trend_y, color="C3", linewidth=1.5, zorder=3)
    ax.scatter(baseline_average_x_date, baseline_average_y, color="C1", s=10.0, zorder=4)
    ax.scatter(baseline_average_x_date, baseline_average_y, color="white", s=4.0, zorder=5)
    for index, (x, y) in enumerate(zip(baseline_average_x_date, baseline_average_y)):
        label = ("$p_{%d}$=%.1f" % (index+1, y)).replace(".", ",")
        ax.annotate(label, xy=(x + datetime.timedelta(days=(index-3) * 80), y-130-3*abs(index-4)), textcoords='data', fontsize=7, horizontalalignment="center", fontweight="semibold")
    if acm_estimate_x_date is not None and acm_estimate_y is not None:
        ax.scatter(acm_estimate_x_date[1:], acm_estimate_y[1:], color="black", s=10.0, zorder=4)
        ax.scatter(acm_estimate_x_date[1:], acm_estimate_y[1:], color="#aaaaaa", s=4.0, zorder=5)
        for index, (x, y) in enumerate(zip(acm_estimate_x_date[1:], acm_estimate_y[1:])):
            label = ("$p_{v%s}$=%.1f" % (index+1, y,)).replace(".", ",")
            ax.annotate(label, xy=(x, y-110), textcoords='data', fontsize=7, horizontalalignment="center", fontweight="semibold")
    #plt.xticks(rotation=45)
    fig.autofmt_xdate(rotation=45, ha="right", which="both")
    plt.grid(True)
    ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
    ax.tick_params(axis="x", pad=0.5, labelsize=7, labelrotation=60)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment('center')
    fig.subplots_adjust(left=0.04, right=0.99, top=0.975, bottom=0.21)
    plot_model_cutoff(fig, ax)
    save_fig(fig, "figures/baseline_trend")
    #plt.show(block=True)
    plt.close(fig)

def plot_acm_baseline_fn(acm_raw_x, acm_raw_x_date, acm_raw_y,
                         acm_averaged_x, acm_averaged_x_date, acm_averaged_y,
                         baseline_average_x, baseline_average_x_date, baseline_average_y,
                         acm_estimate_x, acm_estimate_x_date, acm_estimate_y,
                         baseline_fn, auto_limits):
    fig, ax = plt.subplots(1, 1, figsize=(PAPER_WIDTH_IN, BASELINE_PLOT_HEIGHT))
    #plt.xlim(min(acm_raw_x_date), max(acm_raw_x_date))
    plt.xlim(BASELINE_PLOT_START_DATE, BASELINE_PLOT_END_DATE)
    if not auto_limits:
        ax.set_ylim(ACM_CUSTOM_YLIM)
    (
        baseline_part_x, baseline_part_x_date, baseline_part_y,
        nonbaseline_part_x, nonbaseline_part_x_date, nonbaseline_part_y,
    ) = split_acm_by_cutoff_date(acm_averaged_x, acm_averaged_x_date, acm_averaged_y)
    ax.plot(baseline_part_x_date, baseline_part_y, zorder=1, **BASELINE_FITTING_LINE_PROPS)
    ax.plot(nonbaseline_part_x_date, nonbaseline_part_y, zorder=1, **BASELINE_NONFITTING_LINE_PROPS)

    baseline_range = pandas.date_range(datetime.date(1980, 1, 1), datetime.date(2030, 1, 1), periods=3000)
    baseline_fn_x_date = numpy.array(pandas.to_datetime(baseline_range))
    baseline_fn_x = map_datetime64_to_x(baseline_fn_x_date)
    baseline_fn_y = baseline_fn(baseline_fn_x)
    output_dataseries("data_output/baseline_fn.csv", 
                      ["baseline_fn_x", "baseline_fn_x_date", "baseline_fn_y"], 
                      baseline_fn_x, baseline_fn_x_date, baseline_fn_y)
    ax.plot(baseline_fn_x_date, baseline_fn_y, color="C3", linewidth=DEFAULT_LINEWIDTH, linestyle="-")
    #plt.scatter(baseline_fn_x_date, baseline_fn_y, color="orange", s=3.0)

    fig.autofmt_xdate(rotation=45, ha="right", which="both")
    plt.grid(True)
    ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
    ax.tick_params(axis="x", pad=0.5, labelsize=7, labelrotation=60)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment('center')
    fig.subplots_adjust(left=0.04, right=0.99, top=0.975, bottom=0.21)
    plot_model_cutoff(fig, ax)
    save_fig(fig, "figures/baseline_fn")
    #plt.show(block=True)
    plt.close(fig)

def plot_acm_baseline_trend_and_fn(acm_raw_x, acm_raw_x_date, acm_raw_y,
                                    acm_averaged_x, acm_averaged_x_date, acm_averaged_y,
                                    baseline_average_x, baseline_average_x_date, baseline_average_y,
                                    acm_estimate_x, acm_estimate_x_date, acm_estimate_y,
                                    baseline_trend_fn, baseline_fn, auto_limits):
    fig, ax = plt.subplots(1, 1, figsize=(PAPER_WIDTH_IN, BASELINE_PLOT_HEIGHT))
    #ax.plot(deaths_x, deaths_y, color="red", linewidth=DEFAULT_LINEWIDTH, linestyle="-")
    #plt.scatter(deaths_x, deaths_y, color="red", s=0.5)
    #plt.xlim(min(acm_raw_x_date), max(acm_raw_x_date))
    plt.xlim(BASELINE_PLOT_START_DATE, BASELINE_PLOT_END_DATE)
    if not auto_limits:
        plt.ylim(*ACM_CUSTOM_YLIM)
    (
        baseline_part_x, baseline_part_x_date, baseline_part_y,
        nonbaseline_part_x, nonbaseline_part_x_date, nonbaseline_part_y,
    ) = split_acm_by_cutoff_date(acm_averaged_x, acm_averaged_x_date, acm_averaged_y)
    ax.plot(baseline_part_x_date, baseline_part_y, zorder=1, **BASELINE_FITTING_LINE_PROPS)
    ax.plot(nonbaseline_part_x_date, nonbaseline_part_y, zorder=1, **BASELINE_NONFITTING_LINE_PROPS)
    #ax.plot(baseline_average_x_date, baseline_average_y, color="olivedrab", linewidth=1, linestyle="-", zorder=2)

    baseline_range = pandas.date_range(datetime.date(1980, 1, 1), datetime.date(2030, 1, 1), periods=3000)
    #baseline_range = pandas.date_range(min(acm_raw_x), max(acm_raw_x), periods=200)
    baseline_trend_x_date = numpy.array(pandas.to_datetime(baseline_range))
    baseline_trend_x = map_datetime64_to_x(baseline_trend_x_date)
    baseline_trend_y = baseline_trend_fn(baseline_trend_x)
    ax.plot(baseline_trend_x_date, baseline_trend_y, color="C3", linewidth=1.5, zorder=3)
    if False:
        plt.scatter(baseline_average_x_date, baseline_average_y, color="C1", s=10.0, zorder=4)
        plt.scatter(baseline_average_x_date, baseline_average_y, color="white", s=4.0, zorder=5)
        if False:
            arrowprops = None
        else:
            arrowprops = {
                "connectionstyle": "arc3", 
                "arrowstyle": "-",
                "linewidth": 0.25,
                "edgecolor": "black", 
                #"facecolor": "black",
                "shrinkA": 0.25,
                "shrinkB": 0.25,
            }
        for index, (x, y) in enumerate(zip(baseline_average_x_date, baseline_average_y)):
            label = ("$p_{%d}$=%.1f" % (index+1, y)).replace(".", ",")
            ax.annotate(label, xy=(x, y), xytext=(x + datetime.timedelta(days=70), y-180), 
                        textcoords='data', fontsize=6, horizontalalignment="left", arrowprops=arrowprops)
        if acm_estimate_x_date is not None and acm_estimate_y is not None:
            ax.scatter(acm_estimate_x_date[1:], acm_estimate_y[1:], color="black", s=10.0, zorder=4)
            ax.scatter(acm_estimate_x_date[1:], acm_estimate_y[1:], color="#aaaaaa", s=4.0, zorder=5)
            for index, (x, y) in enumerate(zip(acm_estimate_x_date[1:], acm_estimate_y[1:])):
                label = ("$p_{v%d}$=%.1f" % (index+1, y,)).replace(".", ",")
                ax.annotate(label, xy=(x, y), xytext=(x+datetime.timedelta(days=50), y-180), textcoords='data', fontsize=6, horizontalalignment="left", arrowprops=arrowprops)
    #plt.xticks(rotation=45)
    fig.autofmt_xdate(rotation=45, ha="right", which="both")
    plt.grid(True)
    ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
    ax.tick_params(axis="x", pad=2.5, labelsize=7, labelrotation=None)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment('center')
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(100))

    baseline_range = pandas.date_range(datetime.date(1980, 1, 1), datetime.date(2030, 1, 1), periods=3000)
    baseline_fn_x_date = numpy.array(pandas.to_datetime(baseline_range))
    baseline_fn_x = map_datetime64_to_x(baseline_fn_x_date)
    baseline_fn_y = baseline_fn(baseline_fn_x)
    ax.plot(baseline_fn_x_date, baseline_fn_y, color="C4", linewidth=DEFAULT_LINEWIDTH, linestyle="-")
    #plt.scatter(baseline_fn_x_date, baseline_fn_y, color="orange", s=3.0)

    fig.subplots_adjust(left=0.04, right=0.995, top=0.965, bottom=0.11)
    plot_model_cutoff(fig, ax, ypos=0.11)
    save_fig(fig, "figures/baseline_trend_and_fn")
    #plt.show(block=True)
    plt.close(fig)

def plot_combined_baseline_subplots(acm_raw_x, acm_raw_x_date, acm_raw_y,
                                    acm_averaged_x, acm_averaged_x_date, acm_averaged_y,
                                    baseline_average_x, baseline_average_x_date, baseline_average_y,
                                    acm_estimate_x, acm_estimate_x_date, acm_estimate_y,
                                    baseline_trend_fn, baseline_fn, auto_limits):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(PAPER_WIDTH_IN, 2.0))
    #ax1.plot(deaths_x, deaths_y, color="red", linewidth=DEFAULT_LINEWIDTH, linestyle="-")
    #ax1.scatter(deaths_x, deaths_y, color="red", s=0.5)
    #ax1.xlim(min(acm_raw_x_date), max(acm_raw_x_date))
    ax1.set_xlim(BASELINE_PLOT_START_DATE, BASELINE_PLOT_END_DATE)
    if not auto_limits:
        ax1.set_ylim(*ACM_CUSTOM_YLIM)
    (
        baseline_part_x, baseline_part_x_date, baseline_part_y,
        nonbaseline_part_x, nonbaseline_part_x_date, nonbaseline_part_y,
    ) = split_acm_by_cutoff_date(acm_averaged_x, acm_averaged_x_date, acm_averaged_y)
    acm_avg_line_params = {
        "linewidth": DEFAULT_LINEWIDTH,
        "linestyle": "solid",
        "color": "C0",
    }
    acm_7wkavg_line, = ax1.plot(baseline_part_x_date, baseline_part_y, zorder=1, label="7 viikon juokseva keskiarvo", **acm_avg_line_params)
    acm_nonbaseline_7wkavg_line, = ax1.plot(nonbaseline_part_x_date, nonbaseline_part_y, zorder=1, label="Keskikäyrään vaikuttamaton data", **acm_avg_line_params)
    #ax1.plot(baseline_average_x_date, baseline_average_y, color="olivedrab", linewidth=1, linestyle="-", zorder=2)

    baseline_range = pandas.date_range(datetime.date(1980, 1, 1), datetime.date(2030, 1, 1), periods=3000)
    #baseline_range = pandas.date_range(min(acm_raw_x), max(acm_raw_x), periods=200)
    baseline_trend_x_date = numpy.array(pandas.to_datetime(baseline_range))
    baseline_trend_x = map_datetime64_to_x(baseline_trend_x_date)
    baseline_trend_y = baseline_trend_fn(baseline_trend_x)
    baseline_trend_line, = ax1.plot(baseline_trend_x_date, baseline_trend_y, color="C3", linewidth=1.5, zorder=3,
                                     label="Trendi")
    ax1.scatter(baseline_average_x_date, baseline_average_y, color="C1", s=10.0, zorder=4)
    ax1.scatter(baseline_average_x_date, baseline_average_y, color="white", s=4.0, zorder=5)
    for index, (x, y) in enumerate(zip(baseline_average_x_date, baseline_average_y)):
        label = ("$p_{%d}$=%.1f" % (index+1, y)).replace(".", ",")
        ax1.annotate(label, xy=(x + datetime.timedelta(days=(index-3) * 80), y-145), 
                     textcoords='data', fontsize=7, horizontalalignment="center")
    if acm_estimate_x_date is not None and acm_estimate_y is not None:
        ax1.scatter(acm_estimate_x_date[1:], acm_estimate_y[1:], color="black", s=10.0, zorder=4)
        ax1.scatter(acm_estimate_x_date[1:], acm_estimate_y[1:], color="#aaaaaa", s=4.0, zorder=5)
        for x, y in zip(acm_estimate_x_date[1:], acm_estimate_y[1:]):
            label = ("$p_{ve}$=%.1f" % (y,)).replace(".", ",")
            ax1.annotate(label, xy=(x, y-110), textcoords='data', fontsize=7, horizontalalignment="center", fontweight="semibold")
    #ax1.xticks(rotation=45)
    fig.autofmt_xdate(rotation=45, ha="right", which="both")
    ax1.grid(True)
    ax1.xaxis.set_major_locator(matplotlib.dates.YearLocator())
    ax1.tick_params(axis="x", pad=0.5, labelsize=7, labelrotation=60)
    for label in ax1.get_xticklabels():
        label.set_horizontalalignment('center')

    ax2.set_xlim(BASELINE_PLOT_START_DATE, BASELINE_PLOT_END_DATE)
    ax2.set_ylim(*ax1.get_ylim())

    baseline_fn_x_date = numpy.array(pandas.to_datetime(baseline_range))
    baseline_fn_x = map_datetime64_to_x(baseline_fn_x_date)
    baseline_fn_y = baseline_fn(baseline_fn_x)
    acm_part_baseline_fn_x = map_datetime_to_x(acm_averaged_x_date)
    acm_part_baseline_fn_y = baseline_fn(acm_part_baseline_fn_x)
    ax2.plot(acm_averaged_x_date, acm_averaged_y, color="black", linewidth=0.2, linestyle="solid", zorder=4)
    baseline_fn_line, = ax2.plot(baseline_fn_x_date, baseline_fn_y, color="C3", 
                                 linewidth=0.4, linestyle="solid",
                                 label="Malli $f_{b}(t)$", zorder=3)
    fill_params = {
        "linewidth": 0.05,
        "interpolate": True,
        "alpha": 0.8,
        "edgecolor": "black",
    }
    pos_mortality_fill = ax2.fill_between(acm_averaged_x_date, acm_averaged_y, acm_part_baseline_fn_y,
                                          where=acm_averaged_y > acm_part_baseline_fn_y,
                                          facecolor="C1", label="Ylikuolleisuus", **fill_params)
    neg_mortality_fill = ax2.fill_between(acm_averaged_x_date, acm_averaged_y, acm_part_baseline_fn_y,
                                          where=acm_averaged_y < acm_part_baseline_fn_y, 
                                          facecolor="C6", label="Alikuolleisuus", **fill_params)
    ax2.grid(True)
    ax2.xaxis.set_major_locator(matplotlib.dates.YearLocator())
    ax2.tick_params(axis="x", pad=0.5, labelsize=7, labelrotation=60)
    for label in ax2.get_xticklabels():
        label.set_horizontalalignment('center')
    fig.legend(handles=[acm_7wkavg_line, 
                        #acm_nonbaseline_7wkavg_line, 
                        baseline_trend_line, 
                        baseline_fn_line,
                        neg_mortality_fill,
                        pos_mortality_fill,
                        ], 
               loc=(0.055, 0.51), ncol=5, fontsize=7,
               borderpad=0.3, handletextpad=0.25, labelspacing=0.4, frameon=False, fancybox=False, 
               shadow=False, facecolor="white", framealpha=1.0)
    ax2.annotate('Anomalia', xy=(datetime.date(2022, 1, 1), 1100), xycoords='data',
                 xytext=(0.965, 0.35), textcoords='axes fraction', 
                 arrowprops={ "width": 0.25, "shrink": 0, "connectionstyle": "arc3", "headlength": 2.0, "headwidth": 2.5, "edgecolor": "none", "facecolor": "black", },
                 horizontalalignment='center', verticalalignment='top', fontsize=7)
    fig.text(0.01, 0.5, "Viikoittain kuolleiden lkm", rotation="vertical", 
             verticalalignment="center", horizontalalignment="center", 
             fontsize=7, transform=fig.transFigure)
    fig.subplots_adjust(left=0.055, right=0.99, top=0.985, bottom=0.12, wspace=0, hspace=0.26)
    plot_model_cutoff(fig, ax1)
    plot_model_cutoff(fig, ax2)
    save_fig(fig, "figures/combined_baseline_plots")
    #plt.show(block=True)
    plt.close(fig)

def plot_excess_mortality(excess_mortality_x, excess_mortality_x_date, excess_mortality_y):
    fig, ax = plt.subplots(1, 1, figsize=(PAPER_WIDTH_IN, BASELINE_PLOT_HEIGHT))
    ax.set_xlim(BASELINE_PLOT_START_DATE, BASELINE_PLOT_END_DATE)
    ax.set_ylim(-180, 280)
    acm_avg_line_params = {
        "linewidth": DEFAULT_LINEWIDTH,
        "linestyle": "solid",
        "label": "7 viikon juokseva keskiarvo",
        "color": "C0",
    }
    fig.autofmt_xdate(rotation=45, ha="right", which="both")
    fill_params = {
        "linewidth": 0.05,
        "interpolate": True,
        "alpha": 0.8,
        #"edgecolor": "black",
    }
    pos_mortality_fill = ax.fill_between(excess_mortality_x_date, excess_mortality_y, 0,
                                         where=excess_mortality_y > 0,
                                         facecolor=EXCESS_MORTALITY_FILL_COLOR, label="Ylikuolleisuus", **fill_params)
    neg_mortality_fill = ax.fill_between(excess_mortality_x_date, excess_mortality_y, 0,
                                         where=excess_mortality_y <= 0, 
                                         facecolor=LOW_MORTALITY_FILL_COLOR, label="Alikuolleisuus", **fill_params)
    ax.grid(True)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(100))
    ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
    ax.tick_params(axis="y", labelleft=True, labelright=True, left=True, right=True)
    ax.tick_params(axis="x", pad=2.5, labelsize=7, labelrotation=None)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment('center')
    ax.axhline(linewidth=0.5, color="black")
    plot_rect = {
        "left": 0.04, 
        "right": 0.86,
        "top": 0.985,
        "bottom": 0.12,
    }
    fig.subplots_adjust(**plot_rect)
    ymax = numpy.max(numpy.abs(excess_mortality_y))
    binwidth = 20
    lim = (int(ymax/binwidth) + 1) * binwidth
    bins = numpy.arange(-lim, lim + binwidth, binwidth)
    h_margin = 0.05
    hist_rect = [plot_rect["right"]+h_margin,                # left
                 plot_rect["bottom"],                        # bottom
                 1-plot_rect["right"]-h_margin-0.01,         # width
                 plot_rect["top"]-plot_rect["bottom"],       # height
                 ]
    ax_histy = fig.add_axes(hist_rect, sharey=ax)
    ax_histy.xaxis.set_ticklabels([])
    ax_histy.hist(excess_mortality_y, bins=bins, density=True, orientation='horizontal')
    ax_histy.tick_params(axis="both", top=False, bottom=False, left=False, right=False, labelleft=False, labelright=False)
    yticklabels = list(ax.get_yticklabels())
    assert len(yticklabels) % 2 == 0, "Invalid yticklabels: %s" % (repr(yticklabels),)
    for label in yticklabels[:len(yticklabels)//2]:
        label.set_horizontalalignment('right')
    for label in yticklabels[len(yticklabels)//2:]:
        xy = label.get_position()
        label.set_horizontalalignment('left')
    plot_model_cutoff(fig, ax)
    save_fig(fig, "figures/excess_mortality_raw")
    #plt.show(block=True)
    plt.close(fig)

    excess_mortality_avg_x = excess_mortality_x
    excess_mortality_avg_x_date = excess_mortality_x_date
    excess_mortality_avg_y = calculate_moving_average(excess_mortality_y, 7)
    fig2, ax2 = plt.subplots(1, 1, figsize=(PAPER_WIDTH_IN, BASELINE_PLOT_HEIGHT))
    ax2.set_xlim(BASELINE_PLOT_START_DATE, BASELINE_PLOT_END_DATE)
    ax2.set_ylim(-120, 200)
    acm_avg_line_params = {
        "linewidth": DEFAULT_LINEWIDTH,
        "linestyle": "solid",
        "label": "Kuolleisuus (juokseva keskiarvo)",
        "color": "C0",
    }
    fig2.autofmt_xdate(rotation=45, ha="right", which="both")
    fill_params = {
        "linewidth": 0.05,
        "interpolate": True,
        "alpha": 0.8,
        #"edgecolor": "black",
    }
    pos_mortality_fill = ax2.fill_between(excess_mortality_avg_x_date, excess_mortality_avg_y, 0,
                                          where=excess_mortality_avg_y > 0,
                                          facecolor=EXCESS_MORTALITY_FILL_COLOR, label="Ylikuolleisuus", **fill_params)
    neg_mortality_fill = ax2.fill_between(excess_mortality_avg_x_date, excess_mortality_avg_y, 0,
                                          where=excess_mortality_avg_y <= 0, 
                                          facecolor=LOW_MORTALITY_FILL_COLOR, label="Alikuolleisuus", **fill_params)
    ax2.grid(True)
    ax2.xaxis.set_major_locator(matplotlib.dates.YearLocator())
    ax2.tick_params(axis="x", pad=0.5, labelsize=7, labelrotation=60)
    for label in ax2.get_xticklabels():
        label.set_horizontalalignment('center')
    ax2.axhline(linewidth=0.5, color="black")
    plot_rect = {
        "left": 0.04, 
        "right": 0.86,
        "top": 0.95,
        "bottom": 0.21,
    }
    fig2.subplots_adjust(**plot_rect)
    ymax = numpy.max(numpy.abs(excess_mortality_avg_y))
    binwidth = 10
    lim = (int(ymax/binwidth) + 1) * binwidth
    bins = numpy.arange(-lim, lim + binwidth, binwidth)
    h_margin = 0.05
    hist_rect = [plot_rect["right"]+h_margin,                # left
                 plot_rect["bottom"],                        # bottom
                 1-plot_rect["right"]-h_margin-0.01,         # width
                 plot_rect["top"]-plot_rect["bottom"],       # height
                 ]
    ax2_histy = fig2.add_axes(hist_rect, sharey=ax2)
    ax2_histy.xaxis.set_ticklabels([])
    ax2_histy.hist(excess_mortality_avg_y, bins=bins, density=True, orientation='horizontal')
    plot_model_cutoff(fig, ax2)
    save_fig(fig2, "figures/excess_mortality")
    #plt.show(block=True)
    plt.close(fig2)

def plot_yearly_cumulative_mortality(all_cause_mortality, baseline_fn, covid_data, start_week):
    acm_lookup = dict(((date, deaths) for date, deaths in all_cause_mortality))
    covid_data_lookup = dict(((x[0], x) for x in covid_data))
    fig, ax = plt.subplots(1, 1, figsize=(COLUMN_WIDTH_IN, 2.3))
    plt.ylim(-1800, 4000)
    plt.xlim(1, 52)
    week_labels = []
    for year_idx, year in enumerate(range(BASELINE_START_DATE.year, 2022)):
        cumulative_deaths_x_list = []
        cumulative_deaths = 0
        cumulative_deaths_ex_covid = 0
        cumulative_deaths_y_list = []
        cumulative_deaths_ex_covid_y_list = []
        any_covid_deaths = False
        for week_index in range(0, 52):
            week = week_index + start_week
            if week > 52:
                week = week - 52
                year_incr = 1
            else:
                year_incr = 0
            if year_idx == 0:
                week_labels.append(str(week))
            x_date = get_date_from_isoweek(year+year_incr, week)
            deaths = acm_lookup.get(x_date)
            if deaths is None:
                continue
            cumulative_deaths_x_list.append(week_index + 1)
            x = (x_date - T0_DATE).days
            baseline_deaths = baseline_fn(numpy.array([x]))[0]
            excess_deaths = deaths - baseline_deaths
            cumulative_deaths += excess_deaths
            cumulative_deaths_y_list.append(cumulative_deaths)
            covid_data_item = covid_data_lookup.get(x_date)
            if covid_data_item:
                any_covid_deaths = True
                covid_deaths = covid_data_item[1]
            else:
                covid_deaths = 0
            excess_deaths_ex_covid = excess_deaths - covid_deaths
            cumulative_deaths_ex_covid += excess_deaths_ex_covid
            cumulative_deaths_ex_covid_y_list.append(cumulative_deaths_ex_covid)
        cumulative_deaths_x = numpy.array(cumulative_deaths_x_list)
        cumulative_deaths_y = numpy.array(cumulative_deaths_y_list)
        cumulative_deaths_ex_covid_y = numpy.array(cumulative_deaths_ex_covid_y_list)
        if year < 2000:
            color = "C%d" % (year_idx % 10,)
            linestyle = "dotted"
            linewidth = 0.75
        elif year < 2010:
            color = "C%d" % (year_idx % 10,)
            linestyle = "dashed"
            linewidth = 0.5
        elif year < 2020:
            color = "C%d" % (year_idx % 10,)
            linestyle = "solid"
            linewidth = 0.5
        else:
            color = "C%d" % ((Y2020_COLOR_OFFSET + year - 2020) % 10,)
            linestyle = "solid"
            linewidth = 1.5
        label = str(year)
        ax.plot(cumulative_deaths_x, cumulative_deaths_y, color=color, 
                 linewidth=linewidth, linestyle=linestyle, label=label)
        if year >= 2020 and any_covid_deaths:
            covid_label = "%s*" % (label,)
            ax.plot(cumulative_deaths_x, cumulative_deaths_ex_covid_y, color=color, 
                     linewidth=1.2, linestyle="dashed", label=covid_label)
    ax.legend(loc="upper left", ncol=1, bbox_to_anchor=(1.02,1), fontsize=7, borderpad=0.25, labelspacing=0.135, 
              frameon=True, fancybox=False, shadow=False, facecolor="white", framealpha=1.0, edgecolor="0.5",
              borderaxespad=0)
    fig.text(0.85, 0.12, # 0.91, 0.10, 
             "*) korona-\n    kuolemat\n    vähennetty", 
             horizontalalignment="left", fontsize=6, transform=fig.transFigure)
    week_num = start_week
    week_pos = 1
    xticks = []
    while week_pos <= 52:
        if (week_pos == 1 or (week_num % 5 == 0) or week_num == 1) and week_pos >= 4:
            xticks.append(week_pos)
        week_pos += 1
        week_num = week_num + 1
        if week_num > 52:
            week_num = week_num - 52
                                                                               
    #ax.set_xticks(xticks)
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(xticks))
    def _major_formatter(x, pos):
        return str((int(x) - 1 + start_week - 1) % 52 + 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_major_formatter))
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(500))
    ax.text(0.00, -0.02, "Viikko", fontsize=7, horizontalalignment="right", verticalalignment="top", transform=ax.transAxes)
    #ax.set_xlabel('Viikko')
    ax.axhline(linewidth=0.8, color="black", zorder=0)
    #ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(10))
    # fig.subplots_adjust(left=0.045, right=0.895, top=0.98, bottom=0.06)
    fig.subplots_adjust(left=0.095, right=0.80, top=0.98, bottom=0.06)
    save_fig(fig, "figures/yearly_cumulative_excess_mortality_start_%d" % (start_week,))
    #plt.show(block=True)
    plt.close(fig)

def plot_all_time_cumulative_excess_mortality(all_cause_mortality, baseline_fn):
    cumulative_excess_mortality_x_list = []
    cumulative_excess_mortality_x_date_list = []
    cumulative_excess_mortality_y_list = []
    cumulative_deaths = 0
    for x_date, deaths in all_cause_mortality:
        if x_date < BASELINE_START_DATE:
            continue
        x = (x_date - T0_DATE).days
        cumulative_excess_mortality_x_list.append(x)
        cumulative_excess_mortality_x_date_list.append(x_date)
        baseline_deaths = baseline_fn(numpy.array([x]))[0]
        excess_deaths = deaths - baseline_deaths
        cumulative_deaths += excess_deaths
        cumulative_excess_mortality_y_list.append(cumulative_deaths)
    cumulative_excess_mortality_x = numpy.array(cumulative_excess_mortality_x_list)
    cumulative_excess_mortality_x_date = numpy.array(cumulative_excess_mortality_x_date_list)
    cumulative_excess_mortality_y = numpy.array(cumulative_excess_mortality_y_list)
    output_dataseries("data_output/all_time_cumulative_excess_mortality.csv", 
                      ["cumulative_excess_mortality_x", "cumulative_excess_mortality_x_date", "cumulative_excess_mortality_y"],
                      cumulative_excess_mortality_x, cumulative_excess_mortality_x_date, cumulative_excess_mortality_y)
    fig, ax = plt.subplots(1, 1, figsize=(COLUMN_WIDTH_IN, 2))
    plt.ylim(-1500, 4000)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(500))
    plt.xlim(min(cumulative_excess_mortality_x_date) - datetime.timedelta(days=160), max(cumulative_excess_mortality_x_date) + datetime.timedelta(days=int(0.8*365)))
    fig.autofmt_xdate(rotation=45, ha="right", which="both")
    plt.grid(True)
    ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
    ax.axhline(linewidth=1, color="black")
    ax.tick_params(axis="x", pad=0.5, labelsize=7, labelrotation=60)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment('center')
    ax.plot(cumulative_excess_mortality_x_date, cumulative_excess_mortality_y, color="C1", linewidth=0.5, linestyle="-")
    ax.plot(cumulative_excess_mortality_x_date[-1], cumulative_excess_mortality_y[-1], 'o', ms=20, markerfacecolor='#ffaaaa', markeredgewidth=0, alpha=0.4)
    #plt.scatter(baseline_fn_x_date, baseline_fn_y, color="orange", s=3.0)
    fig.subplots_adjust(bottom=0.11, top=0.975, left=0.095, right=0.99)
    plot_model_cutoff(fig, ax, ypos=0.09)
    save_fig(fig, "figures/all_time_cumulative_excess_mortality")
    #plt.show(block=True)

    fig.subplots_adjust(bottom=0.11, top=0.92, left=0.095, right=0.99)
    ax_ylabel_transform = matplotlib.transforms.blended_transform_factory(fig.transFigure, ax.transAxes)
    ax.text(0.5, 1.02, "Kumuloitu ylikuolleisuus", transform=ax_ylabel_transform, fontsize=9,
            horizontalalignment="center", verticalalignment="bottom")
    #ax.text(0.02, 0.5, "Kumulatiivinen ylikuolleisuus", transform=ax_ylabel_transform, fontsize=7, rotation=90,
    #        horizontalalignment="center", verticalalignment="center")
    #ax.text(0.005, -0.05, "Vuosi", transform=ax_ylabel_transform, fontsize=7,
    #        horizontalalignment="left", verticalalignment="top")
    save_fig(fig, "figures/all_time_cumulative_excess_mortality_labels")

    plt.close(fig)

def plot_covid_cases_and_deaths(covid_data):
    covid_x_list = []
    covid_x_date_list = []
    covid_deaths_y_list = []
    covid_cases_y_list = []
    covid_tests_y_list = []
    summer_2021_covid_deaths_sum = 0
    for x_date, covid_deaths, covid_cases, covid_tests in covid_data:
        if x_date < datetime.date(2021, 1, 1):
            continue
        x = (x_date - T0_DATE).days
        covid_x_list.append(x)
        covid_x_date_list.append(x_date)
        covid_deaths_y_list.append(covid_deaths)
        covid_cases_y_list.append(covid_cases)
        covid_tests_y_list.append(covid_tests)
        if x_date >= datetime.date(2021, 6, 1) and x_date < datetime.date(2021, 8, 1):
            summer_2021_covid_deaths_sum += covid_deaths
    covid_x = numpy.array(covid_x_list)
    covid_x_date = numpy.array(covid_x_date_list)
    covid_deaths_y = numpy.array(covid_deaths_y_list)
    covid_cases_y = numpy.array(covid_cases_y_list)
    covid_tests_y = numpy.array(covid_tests_y_list)
    print("Summer 2021 covid deaths: %d" % (summer_2021_covid_deaths_sum,))

    fig1, ax1 = plt.subplots(1, 1, figsize=(COLUMN_WIDTH_IN, 0.8))
    ax1.set_ylim(0, 120)
    ax1.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(20))
    fig1.autofmt_xdate(rotation=0, ha="right", which="both")
    ax1.set_xlim(datetime.date(2021, 1, 1), datetime.date(2022, 1, 1))
    ax1.xaxis.set_major_locator(matplotlib.dates.YearLocator())
    ax1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    ax1.xaxis.set_minor_locator(matplotlib.dates.MonthLocator())
    ax1.xaxis.set_minor_formatter(matplotlib.dates.DateFormatter("%m"))
    for label in ax1.get_xticklabels(which="both"):
        label.set_horizontalalignment('center')
    deaths_line, = ax1.plot(covid_x_date, covid_deaths_y, color="C1", linewidth=DEFAULT_LINEWIDTH, linestyle="solid", 
                            label="Koronaan kuolleet")
    ax1.scatter(covid_x_date, covid_deaths_y, color="C1", s=1.0)
    ax1.tick_params(axis="x", which="minor", reset=True, pad=2, labelsize=7, labelrotation=0, direction="out", 
                    top=False, bottom=True, labeltop=False, labelbottom=True)
    ax1.tick_params(axis="x", which="major",
                    top=False, bottom=True, labeltop=False, labelbottom=False)
    ax1.tick_params(axis="y", direction="out", reset=False, left=True, right=False, which="both")
    #legend = plt.legend(handles=[deaths_line], loc='upper left',
    #                    bbox_to_anchor=(0.01, 0.99), fontsize=6,
    #                    borderpad=0.3, labelspacing=0.2, frameon=True, fancybox=False, 
    #                    shadow=False, facecolor="white", framealpha=1.0, edgecolor="black")
    fig1.subplots_adjust(left=0.08, right=0.985, top=0.955, bottom=0.165)
    save_fig(fig1, "figures/covid_deaths")
    #legend.remove()

    ax1b = ax1.twinx()
    ax1b.set_ylim(0, 75000)
    ax1b.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10000))
    ax1b.set_xlim(datetime.date(2021, 1, 1), max(covid_x_date) + datetime.timedelta(days=4))
    ax1b.xaxis.set_major_locator(matplotlib.dates.YearLocator())
    ax1b.xaxis.set_minor_locator(matplotlib.dates.MonthLocator())
    cases_line, = ax1b.plot(covid_x_date, covid_cases_y, color="C2", linewidth=DEFAULT_LINEWIDTH, linestyle="solid", 
                            label="Koronatapaukset")
    for label in ax1b.get_xticklabels():
        label.set_horizontalalignment('center')
    ax1b.tick_params(axis="x", pad=2, labelsize=7, labelrotation=0, direction="out", 
                    reset=True, top=False, bottom=True, left=True, right=True,
                    which="both")
    ax1b.tick_params(axis="y", direction="out", reset=False, left=False, right=True, which="both")
    legend = plt.legend(handles=[deaths_line, cases_line], loc='upper left',
                        bbox_to_anchor=(0.01, 0.99), fontsize=6,
                        borderpad=0.3, labelspacing=0.2, frameon=True, fancybox=False, 
                        shadow=False, facecolor="white", framealpha=1.0, edgecolor="black")
    fig1.subplots_adjust(left=0.08, right=0.89, top=0.99, bottom=0.12)
    save_fig(fig1, "figures/covid_cases_and_deaths")
    plt.close(fig1)

def plot_euromomo_zscores(country_euromomo_data, output_file_name, output_cumulative_file_name, output_combined_file_name):
    zscore_x_list = []
    zscore_x_date_list = []
    zscore_y_list = []
    cumulative_zscore_x_list = []
    cumulative_zscore_x_date_list = []
    cumulative_zscore_y_list = []
    cumulative_zscore = 0
    pre_2021_cumulative_max = 0
    for x_date, zscore in country_euromomo_data[:-2]:
        x = (x_date - T0_DATE).days
        zscore_x_list.append(x)
        zscore_x_date_list.append(x_date)
        zscore_y_list.append(zscore)
        if x_date >= datetime.date(2017, 1, 1):
            if len(cumulative_zscore_x_list) == 0:
                prev_x_date = x_date - datetime.timedelta(days=7)
                prev_x = (prev_x_date - T0_DATE).days
                cumulative_zscore_x_list.append(prev_x)
                cumulative_zscore_x_date_list.append(prev_x_date)
                cumulative_zscore_y_list.append(0)
            cumulative_zscore += zscore
            cumulative_zscore_x_list.append(x)
            cumulative_zscore_x_date_list.append(x_date)
            cumulative_zscore_y_list.append(cumulative_zscore)
            if x_date < datetime.date(2021, 1, 1) and cumulative_zscore > pre_2021_cumulative_max:
                pre_2021_cumulative_max = cumulative_zscore
    zscore_x = numpy.array(zscore_x_list)
    zscore_x_date = numpy.array(zscore_x_date_list)
    zscore_y = numpy.array(zscore_y_list)
    zscore_avg_y = calculate_moving_average(zscore_y, 7)
    cumulative_zscore_x = numpy.array(cumulative_zscore_x_list)
    cumulative_zscore_x_date = numpy.array(cumulative_zscore_x_date_list)
    cumulative_zscore_y = numpy.array(cumulative_zscore_y_list)

    fig1, ax1 = plt.subplots(1, 1, figsize=(COLUMN_WIDTH_IN, 1.6))
    ax1.set_xlim(min(zscore_x_date), max(zscore_x_date))
    ax1.axhline(linewidth=0.8, color="black")
    fig1.autofmt_xdate(rotation=0, ha="right", which="both")
    ax1.xaxis.set_major_locator(matplotlib.dates.YearLocator())
    ax1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    ax1.axhline(y=4, linewidth=0.5, color="r", linestyle="dashed")
    ax1.plot(zscore_x_date, zscore_y, linewidth=0.4, color="C5", alpha=0.4)
    ax1.plot(zscore_x_date, zscore_avg_y, linewidth=0.6, color=BLUE_COLOR, linestyle="solid")
    fig1.subplots_adjust(left=0.08, right=0.99, top=0.99, bottom=0.08)
    save_fig(fig1, output_file_name)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(1, 1, figsize=(COLUMN_WIDTH_IN, 1.4))
    fig2.autofmt_xdate(rotation=0, ha="right", which="both")
    # ax2.set_ylim(-50, 850)
    ax2.set_xlim(min(zscore_x_date), max(zscore_x_date))
    ax2.axhline(linewidth=0.8, color="black")
    ax2.xaxis.set_major_locator(matplotlib.dates.YearLocator())
    ax2.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    fill_params = {
        "linewidth": 0.05,
        "interpolate": True,
        "alpha": 0.8,
        "edgecolor": "black",
    }
    ax2.fill_between(cumulative_zscore_x_date, cumulative_zscore_y, 0,
                     where=cumulative_zscore_y > 0,
                     facecolor=EXCESS_MORTALITY_FILL_COLOR, label="Ylikuolleisuus EuroMoMo", **fill_params)
    ax2.fill_between(cumulative_zscore_x_date, cumulative_zscore_y, 0,
                     where=cumulative_zscore_y <= 0, 
                     facecolor=LOW_MORTALITY_FILL_COLOR, label="Alikuolleisuus EuroMoMo", **fill_params)
    fig2.subplots_adjust(left=0.10, right=0.99, top=0.99, bottom=0.08)
    save_fig(fig2, output_cumulative_file_name)
    plt.close(fig2)
    
    fig3, (ax3, ax4) = plt.subplots(2, 1, figsize=(COLUMN_WIDTH_IN, 2.4))
    xlim_start, xlim_end = datetime.date(2017, 1, 1), datetime.date(2022, 12, 31)
    #xlim_start, xlim_end = min(zscore_x_date), max(zscore_x_date)
    fig3.autofmt_xdate(rotation=0, ha="right", which="both")
    # ax3.set_ylim(-50, 850)
    for ax in (ax3, ax4):
        ax.set_xlim(xlim_start, xlim_end)
    ax3.axhline(linewidth=0.4, color="black")
    ax4.grid(axis="y", visible=False)
    hline_xmax = (max(cumulative_zscore_x_date)-xlim_start).days / (xlim_end - xlim_start).days + 0.02
    ax4.axhline(xmax=hline_xmax, linewidth=0.6, color="C3", linestyle="dotted")
    yline_transform = matplotlib.transforms.blended_transform_factory(ax4.transAxes, ax4.transData)
    ax4.text(hline_xmax+0.01, 0, "Tasapaino", 
             transform=yline_transform, fontsize=5, color="C3", rotation="-45", rotation_mode="anchor",
             horizontalalignment="left", verticalalignment="center")
    ax4.axhline(y=pre_2021_cumulative_max, xmax=hline_xmax, linewidth=0.6, color="C3", linestyle="dotted")
    ax4.text(hline_xmax+0.01, pre_2021_cumulative_max, "<2021 maksimi", 
             transform=yline_transform, fontsize=5, color="C3", rotation="-45", rotation_mode="anchor",
             horizontalalignment="left", verticalalignment="center")
    ax4.xaxis.set_minor_locator(WeekNumberLocator([20, 40]))
    ax4.xaxis.set_minor_formatter(WeekNumberFormatter())
    fill_params = {
        "linewidth": 0.05,
        "interpolate": True,
        "alpha": 0.8,
        "edgecolor": "black",
    }
    ax3.scatter(zscore_x_date, zscore_y, s=2, color=BLUE_COLOR)
    ax3.plot(zscore_x_date, zscore_y, linewidth=0.25, color=BLUE_COLOR)
    #ax3.plot(zscore_x_date, zscore_avg_y, linewidth=0.6, color="C6", linestyle="solid")
    ax4.fill_between(cumulative_zscore_x_date, cumulative_zscore_y, 0,
                     where=cumulative_zscore_y > 0,
                     facecolor=EXCESS_MORTALITY_FILL_COLOR, label="Ylikuolleisuus EuroMoMo", **fill_params)
    ax4.fill_between(cumulative_zscore_x_date, cumulative_zscore_y, 0,
                     where=cumulative_zscore_y <= 0, 
                     facecolor=LOW_MORTALITY_FILL_COLOR, label="Alikuolleisuus EuroMoMo", **fill_params)
    # Show year labels in top x-axis
    ax3b = ax3.twiny()
    ax4b = ax4.twiny()
    for ax in (ax3b, ax4b):
        ax.tick_params(axis="x", which="major", reset=True, labelsize=7, pad=0,
                       labeltop=True, labelbottom=False, top=False, bottom=False)
    for ax in (ax3, ax4):
        ax.tick_params(axis="x", which="major", reset=True, labeltop=False, labelbottom=False, top=False, bottom=False)
    ax4.tick_params(axis="x", which="minor", reset=True, labelsize=7, pad=2,
                    labeltop=False, labelbottom=True, top=False, bottom=True)
    for ax in (ax3b, ax4b):
        ax.set_xlim(xlim_start, xlim_end)
        ax.grid(which="both", visible=False)
        ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(month=7))
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
        for label in ax.get_xticklabels(which="major"):
            label.set_horizontalalignment('center')
    # Show week numbers in bottom x-axis
    for ax in (ax3, ax4):
        ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
        ax.grid(axis="x", color='gray', linestyle='solid', linewidth=0.5)
    ax4.grid(axis="x", which="minor", visible=False)
    # Labels manually
    ax3_ylabel_transform = matplotlib.transforms.blended_transform_factory(fig3.transFigure, ax3.transAxes)
    ax3.text(0.02, 0.5, "Viikoittainen z", transform=ax3_ylabel_transform, fontsize=7, rotation=90,
             horizontalalignment="center", verticalalignment="center")
    ax4_ylabel_transform = matplotlib.transforms.blended_transform_factory(fig3.transFigure, ax4.transAxes)
    ax4.text(0.02, 0.5, "Kumulatiivinen z", transform=ax4_ylabel_transform, fontsize=7, rotation=90,
             horizontalalignment="center", verticalalignment="center")
    ax4.text(0.005, -0.02, "Viikko", transform=ax4_ylabel_transform, fontsize=7,
             horizontalalignment="left", verticalalignment="top")
    # Rectangles
    rect_transform = matplotlib.transforms.blended_transform_factory(ax4.transAxes, fig3.transFigure)
    rect_start_x = (datetime.date(2020, 3, 1)-xlim_start).days / (xlim_end - xlim_start).days
    rect_end_x = (datetime.date(2020, 7, 1)-xlim_start).days / (xlim_end - xlim_start).days
    ax4.add_patch(matplotlib.patches.Rectangle((rect_start_x, 0.07), rect_end_x-rect_start_x, 0.8, transform=rect_transform, 
                                               linewidth=DEFAULT_LINEWIDTH, color="C3", fill=False, clip_on=False, zorder=100))
    rect_start_x = (datetime.date(2021, 5, 1)-xlim_start).days / (xlim_end - xlim_start).days
    rect_end_x = (datetime.date(2022, 3, 1)-xlim_start).days / (xlim_end - xlim_start).days
    rect_transform = matplotlib.transforms.blended_transform_factory(ax3.transAxes, ax3.transData)
    rect_patch = matplotlib.patches.Rectangle((rect_start_x, 0.1), rect_end_x-rect_start_x, 3.8, transform=rect_transform, 
                                              linewidth=DEFAULT_LINEWIDTH, color="C3", fill=False, clip_on=False, zorder=100)
    ax4.add_patch(rect_patch)
    fig3.subplots_adjust(left=0.11, right=0.99, top=0.94, bottom=0.06, wspace=0, hspace=0.12)
    save_fig(fig3, output_combined_file_name)
    plt.close(fig3)
    
def plot_highlighted_euromomo_zscores(country_euromomo_data):
    zscore_x_list = []
    zscore_x_date_list = []
    zscore_y_list = []
    cumulative_zscore_x_list = []
    cumulative_zscore_x_date_list = []
    cumulative_zscore_y_list = []
    cumulative_zscore = 0
    pre_2021_cumulative_max = 0
    for x_date, zscore in country_euromomo_data[:-2]:
        x = (x_date - T0_DATE).days
        zscore_x_list.append(x)
        zscore_x_date_list.append(x_date)
        zscore_y_list.append(zscore)
        if x_date >= datetime.date(2017, 1, 1):
            if len(cumulative_zscore_x_list) == 0:
                prev_x_date = x_date - datetime.timedelta(days=7)
                prev_x = (prev_x_date - T0_DATE).days
                cumulative_zscore_x_list.append(prev_x)
                cumulative_zscore_x_date_list.append(prev_x_date)
                cumulative_zscore_y_list.append(0)
            cumulative_zscore += zscore
            cumulative_zscore_x_list.append(x)
            cumulative_zscore_x_date_list.append(x_date)
            cumulative_zscore_y_list.append(cumulative_zscore)
            if x_date < datetime.date(2021, 1, 1) and cumulative_zscore > pre_2021_cumulative_max:
                pre_2021_cumulative_max = cumulative_zscore
    zscore_x = numpy.array(zscore_x_list)
    zscore_x_date = numpy.array(zscore_x_date_list)
    zscore_y = numpy.array(zscore_y_list)
    zscore_avg_y = calculate_moving_average(zscore_y, 7)
    cumulative_zscore_x = numpy.array(cumulative_zscore_x_list)
    cumulative_zscore_x_date = numpy.array(cumulative_zscore_x_date_list)
    cumulative_zscore_y = numpy.array(cumulative_zscore_y_list)

    fig, ax3 = plt.subplots(1, 1, figsize=(COLUMN_WIDTH_IN, 1.2))
    xlim_start, xlim_end = datetime.date(2017, 1, 1), datetime.date(2022, 12, 31)
    #xlim_start, xlim_end = min(zscore_x_date), max(zscore_x_date)
    fig.autofmt_xdate(rotation=0, ha="right", which="both")
    ax3.set_ylim(-3, 4)
    ax3.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
    ax3.set_xlim(xlim_start, xlim_end)
    ax3.axhline(linewidth=0.4, color="black")
    fill_params = {
        "linewidth": 0.05,
        "interpolate": True,
        "alpha": 0.8,
        "edgecolor": "black",
    }
    ax3.scatter(zscore_x_date, zscore_y, s=1, color=BLUE_COLOR)
    ax3.plot(zscore_x_date, zscore_y, linewidth=0.25, color=BLUE_COLOR)
    #ax3.plot(zscore_x_date, zscore_avg_y, linewidth=0.6, color="C6", linestyle="solid")
    # Show year labels in top x-axis
    ax3b = ax3.twiny()
    ax3b.tick_params(axis="x", which="major", reset=True, labelsize=7, pad=0,
                     labeltop=True, labelbottom=False, top=False, bottom=False)
    ax3.tick_params(axis="x", which="major", reset=True, labeltop=False, labelbottom=False, top=False, bottom=False)
    ax3b.set_xlim(xlim_start, xlim_end)
    ax3b.grid(which="both", visible=False)
    ax3b.xaxis.set_major_locator(matplotlib.dates.YearLocator(month=7))
    ax3b.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    for label in ax3b.get_xticklabels(which="major"):
        label.set_horizontalalignment('center')
    # Show week numbers in bottom x-axis
    ax3.xaxis.set_major_locator(matplotlib.dates.YearLocator())
    ax3.grid(axis="x", color='gray', linestyle='solid', linewidth=0.5)
    # Labels manually
    ax3_ylabel_transform = matplotlib.transforms.blended_transform_factory(fig.transFigure, ax3.transAxes)
    ax3.text(0.02, 0.5, "Viikoittainen z", transform=ax3_ylabel_transform, fontsize=7, rotation=90,
             horizontalalignment="center", verticalalignment="center")
    # Rectangles
    rect_start_x = (get_date_from_isoweek(2021, 24)-xlim_start).days / (xlim_end - xlim_start).days
    rect_end_x = (get_date_from_isoweek(2022, 13)-xlim_start).days / (xlim_end - xlim_start).days
    rect_patch = matplotlib.patches.Rectangle((rect_start_x, 0), rect_end_x-rect_start_x, 1, transform=ax3.transAxes, 
                                              linewidth=None, color="#ffaaaa", fill=True, clip_on=True, zorder=0)
    ax3.add_patch(rect_patch)
    fig.subplots_adjust(left=0.08, right=0.995, top=0.92, bottom=0.04, wspace=0, hspace=0.12)
    save_fig(fig, "figures/euromomo_zscores_highlighted")
    plt.close(fig)

def plot_euromomo_vs_model_cumulative(excess_mortality_x, excess_mortality_x_date, excess_mortality_y, country_euromomo_data):
    zscore_x_list = []
    zscore_x_date_list = []
    zscore_y_list = []
    cumulative_zscore_x_list = []
    cumulative_zscore_x_date_list = []
    cumulative_zscore_y_list = []
    cumulative_zscore = 0
    cumulative_excmort = 0
    cumulative_excmort_y_list = []
    prev_x_date = None
    excess_mortality_idx = excess_mortality_x_date.searchsorted(country_euromomo_data[0][0])
    assert excess_mortality_idx is not None and excess_mortality_idx >= 0
    for (x_date, zscore), excmort_x, excmort_x_d, excmort in zip(country_euromomo_data[:-2], 
                                                                 excess_mortality_x[excess_mortality_idx:], 
                                                                 excess_mortality_x_date[excess_mortality_idx:], 
                                                                 excess_mortality_y[excess_mortality_idx:]):
        x = (x_date - T0_DATE).days
        assert excmort_x_d == x_date, "excmort_x_d '%s' != x_date '%s'" % (excmort_x_d, x_date)
        assert excmort_x == x
        zscore_x_list.append(x)
        zscore_x_date_list.append(x_date)
        zscore_y_list.append(zscore)
        if x_date >= datetime.date(2017, 1, 1):
            if len(cumulative_zscore_x_list) == 0:
                prev_x_date = x_date - datetime.timedelta(days=7)
                prev_x = (prev_x_date - T0_DATE).days
                cumulative_zscore_x_list.append(prev_x)
                cumulative_zscore_x_date_list.append(prev_x_date)
                cumulative_zscore_y_list.append(0)
                cumulative_excmort_y_list.append(0)
            cumulative_zscore += zscore
            cumulative_excmort += excmort
            cumulative_zscore_x_list.append(x)
            cumulative_zscore_x_date_list.append(x_date)
            cumulative_zscore_y_list.append(cumulative_zscore)
            cumulative_excmort_y_list.append(cumulative_excmort)
        prev_x_date = x_date
    zscore_x = numpy.array(zscore_x_list)
    zscore_x_date = numpy.array(zscore_x_date_list)
    zscore_y = numpy.array(zscore_y_list)
    zscore_avg_y = calculate_moving_average(zscore_y, 7)
    cumulative_zscore_x = numpy.array(cumulative_zscore_x_list)
    cumulative_zscore_x_date = numpy.array(cumulative_zscore_x_date_list)
    cumulative_zscore_y = numpy.array(cumulative_zscore_y_list)
    cumulative_excmort_y = numpy.array(cumulative_excmort_y_list)

    fig1, ax1 = plt.subplots(1, 1, figsize=(COLUMN_WIDTH_IN, 1.6))
    ax1.set_xlim(min(zscore_x_date), max(zscore_x_date))
    ax1.axhline(linewidth=0.8, color="black")
    fig1.autofmt_xdate(rotation=0, ha="right", which="both")
    ax1.xaxis.set_major_locator(matplotlib.dates.YearLocator())
    ax1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    ax1.axhline(y=4, linewidth=0.5, color="r", linestyle="dashed")
    ax1.plot(zscore_x_date, zscore_y, linewidth=0.4, color="C5", alpha=0.4)
    ax1.plot(zscore_x_date, zscore_avg_y, linewidth=0.6, color="C6", linestyle="solid")
    fig1.subplots_adjust(left=0.08, right=0.99, top=0.99, bottom=0.08)
    save_fig(fig1, "figures/euromomo_zscores")
    plt.close(fig1)

    fig2, (ax2, ax2b) = plt.subplots(2, 1, figsize=(COLUMN_WIDTH_IN, 2.2))
    fig2.autofmt_xdate(rotation=0, ha="right", which="both")
    for ax in (ax2, ax2b):
        ax.set_xlim(min(zscore_x_date), max(zscore_x_date))
        ax.axhline(linewidth=0.8, color="black")
        ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    fill_params = {
        "linewidth": 0.05,
        "interpolate": True,
        "alpha": 0.8,
        "edgecolor": "black",
    }
    ax2.fill_between(cumulative_zscore_x_date, cumulative_zscore_y, 0,
                     where=cumulative_zscore_y > 0,
                     facecolor=EXCESS_MORTALITY_FILL_COLOR, label="Ylikuolleisuus EuroMoMo", **fill_params)
    ax2.fill_between(cumulative_zscore_x_date, cumulative_zscore_y, 0,
                     where=cumulative_zscore_y <= 0, 
                     facecolor=LOW_MORTALITY_FILL_COLOR, label="Alikuolleisuus EuroMoMo", **fill_params)
    ax2b.fill_between(cumulative_zscore_x_date, cumulative_excmort_y, 0,
                      where=cumulative_excmort_y > 0,
                      facecolor=EXCESS_MORTALITY_FILL_COLOR, label="Ylikuolleisuus", **fill_params)
    ax2b.fill_between(cumulative_zscore_x_date, cumulative_excmort_y, 0,
                      where=cumulative_excmort_y <= 0, 
                      facecolor=LOW_MORTALITY_FILL_COLOR, label="Alikuolleisuus", **fill_params)
    fig2.subplots_adjust(left=0.10, right=0.99, top=0.99, bottom=0.08, wspace=0, hspace=0.06)
    save_fig(fig2, "figures/euromomo_vs_model_cumulative")
    plt.close(fig2)

def plot_euromomo_correlation(excess_mortality_x, excess_mortality_x_date, excess_mortality_y, country_euromomo_data):
    excess_mortality_lookup = {}
    for x_date, excess_mortality in zip(excess_mortality_x_date, excess_mortality_y):
        excess_mortality_lookup[x_date] = excess_mortality
    correlation_x_list = []
    correlation_x_date_list = []
    correlation_excess_mortality_list = []
    correlation_zscore_list = []
    year_correlations = []
    year_correllation_lookup = {}
    for x_date, zscore in country_euromomo_data:
        if x_date.year < 2017:
            continue
        if x_date not in excess_mortality_lookup:
            continue
        excess_mortality = excess_mortality_lookup[x_date]
        x = (x_date - T0_DATE).days
        correlation_x_list.append(x)
        correlation_x_date_list.append(x_date)
        correlation_excess_mortality_list.append(excess_mortality)
        correlation_zscore_list.append(zscore)
        year = x_date.year
        if year in year_correllation_lookup:
            year_item = year_correllation_lookup[year]
        else:
            year_item = (len(year_correlations), year, [], [], [], [])
            year_correlations.append(year_item)
            year_correllation_lookup[year] = year_item
        year_item[2].append(x)
        year_item[3].append(x_date)
        year_item[4].append(excess_mortality)
        year_item[5].append(zscore)
    correlation_x = numpy.array(correlation_x_list)
    correlation_x_date = numpy.array(correlation_x_date_list)
    correlation_excess_mortality = numpy.array(correlation_excess_mortality_list)
    correlation_zscore = numpy.array(correlation_zscore_list)

    fig, ax1 = plt.subplots(1, 1, figsize=(COLUMN_WIDTH_IN, 1.3))
    ax1.set_xlim(-4, 5)
    ylim_min, ylim_max = -120, 250
    ax1.set_ylim(ylim_min, ylim_max)
    assert min(correlation_excess_mortality) > ylim_min
    assert max(correlation_excess_mortality) < ylim_max
    ax1.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
    ax1.axhline(linewidth=0.5, color="black")
    ax1.axvline(linewidth=0.5, color="black")
    symbols = "D*Xsov^Ph"
    year_scatters = []
    for year_index, year, year_x_list, year_x_date_list, year_excess_mortality_list, year_zscore_list in year_correlations:
        year_x = numpy.array(year_x_list)
        year_x_date = numpy.array(year_x_date_list)
        year_excess_mortality = numpy.array(year_excess_mortality_list)
        year_zscore = numpy.array(year_zscore_list)
        year_scatters.append(ax1.scatter(year_zscore, year_excess_mortality, s=6.0, label=str(year), alpha=0.8, c="C0", marker=symbols[year_index%len(symbols)]))
    trendline_coeff = numpy.polyfit(correlation_zscore, correlation_excess_mortality, 1)
    trendline_fn = numpy.poly1d(trendline_coeff)
    ax1.plot(correlation_zscore, trendline_fn(correlation_zscore), linewidth=0.4, color="r", linestyle="dashed")
    pearsons_r = numpy.corrcoef(correlation_zscore, correlation_excess_mortality)[1,0]
    ax1.text(0.80, 0.64, "r=%.2f" % (pearsons_r,), fontsize=7, color="r", transform=ax1.transAxes)
    ax1_ylabel_transform = matplotlib.transforms.blended_transform_factory(fig.transFigure, ax1.transAxes)
    ax1.text(0.04, 0.5, "Viikoittainen\nylikuolleisuus", transform=ax1_ylabel_transform, fontsize=7, rotation=90,
             horizontalalignment="center", verticalalignment="center")
    ax1.text(0.02, -0.04, "Z-arvo", transform=ax1_ylabel_transform, fontsize=7, horizontalalignment="left", verticalalignment="top")
    fig.subplots_adjust(left=0.16, right=0.82, top=0.98, bottom=0.1)
    ax1.legend(handles=year_scatters, loc='upper left', ncol=1, bbox_to_anchor=(1.04, 1), fontsize=8,
               handletextpad=0.4, handlelength=0.5, borderpad=0.5, labelspacing=0.3, 
               frameon=True, fancybox=False, shadow=False, facecolor="white", framealpha=1.0, edgecolor="0.5",
               borderaxespad=0)
    save_fig(fig, "figures/euromomo_correlation")
    plt.close(fig)

def plot_processcontrol_deaths_by_halfyears(deaths_and_population_by_month):
    deaths_by_halfyears_x_list = []
    deaths_by_halfyears_x_date_list = []
    deaths_by_halfyears_y_list = []
    dbhy_statistics_y = []
    current_year = None
    current_h1_sum = None
    current_h1_counter = None
    current_h2_sum = None
    current_h2_counter = None
    for month_item in deaths_and_population_by_month:
        year = month_item["year"]
        month = month_item["month"]
        deaths = month_item["deaths"]
        if month == 1:
            if current_year is not None:
                assert current_h1_sum is not None and current_h2_sum is not None
                assert current_h1_sum > 0 and current_h2_sum > 0
                assert current_h1_counter == 6, "Invalid h1 counter: %s" % (current_h1_counter,)
                assert current_h2_counter == 6, "Invalid h2 counter: %s" % (current_h2_counter,)
                data_date = datetime.date(current_year, 1, 1)
                data_x = (data_date - T0_DATE).days
                data_y = current_h2_sum / (current_h1_sum + current_h2_sum)
                deaths_by_halfyears_x_list.append(data_x)
                deaths_by_halfyears_x_date_list.append(data_date)
                deaths_by_halfyears_y_list.append(data_y)
                if data_date < T0_DATE:
                    dbhy_statistics_y.append(data_y)
            current_year = year
            current_h1_sum = 0
            current_h1_counter = 0
            current_h2_sum = None
            current_h2_counter = None
        elif month == 7 and current_h1_sum is not None:
            current_h2_sum = 0
            current_h2_counter = 0
        if 1 <= month < 7:
            assert current_h2_sum is None
            assert current_h2_counter is None
            current_h1_sum += deaths
            current_h1_counter += 1
        else:
            if current_h2_sum is not None:
                current_h2_sum += deaths
                current_h2_counter += 1
    deaths_by_halfyears_x = numpy.array(deaths_by_halfyears_x_list)
    deaths_by_halfyears_x_date = numpy.array(deaths_by_halfyears_x_date_list)
    deaths_by_halfyears_y = numpy.array(deaths_by_halfyears_y_list)
    dbhy_average = numpy.average(dbhy_statistics_y)
    dbhy_stddev = numpy.std(dbhy_statistics_y)

    fig1, ax1 = plt.subplots(1, 1, figsize=(COLUMN_WIDTH_IN, 1.1))
    ax1.set_ylim(0.46, 0.54)
    fig1.autofmt_xdate(rotation=45, ha="right", which="both")
    #ax1.set_ylabel("Toisen vuosipuoliskon kuolemien osuus")
    #ax1.set_xlim(min(deaths_by_halfyears_x_date_list)-datetime.timedelta(days=800), 
    #             max(deaths_by_halfyears_x_date_list)+datetime.timedelta(days=5000))
    ax1.set_xlim(min(deaths_by_halfyears_x_date_list)-datetime.timedelta(days=400), 
                 max(deaths_by_halfyears_x_date_list)+datetime.timedelta(days=2500))
    ax1.xaxis.set_major_locator(matplotlib.dates.YearLocator(5))
    for label in ax1.get_xticklabels():
        label.set_horizontalalignment('center')
    ax1.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.02))
    ax1.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))
    deaths_by_halfyears_scatter = ax1.scatter(deaths_by_halfyears_x_date, deaths_by_halfyears_y, color=BLUE_COLOR, s=6)
    deaths_by_halfyears_line, = ax1.plot(deaths_by_halfyears_x_date, deaths_by_halfyears_y, color=BLUE_COLOR, 
                                         linewidth=0.3, linestyle="solid",)
    ax1.tick_params(axis="x", pad=2.5, labelsize=7, labelrotation=None)
    ax1.tick_params(axis="y", which="major", labelright=True)
    #plt.legend(handles=[deaths_by_halfyears_line,], loc='upper left',
    #                    bbox_to_anchor=(0.01, 0.99), fontsize=6,
    #                    borderpad=0.3, labelspacing=0.2, frameon=True, fancybox=False, 
    #                    shadow=False, facecolor="white", framealpha=1.0, edgecolor="black")
    fig1.subplots_adjust(bottom=0.10, top=0.965, left=0.075, right=0.99)

    for px, py in [(deaths_by_halfyears_x_date[-1], deaths_by_halfyears_y[-1])]:
        p_text = (r"%s: %.1f%% $(%+.1f\sigma)$" % (px.year, 100*py, (py - dbhy_average) / dbhy_stddev)).replace(".", ",")
        p_transform = matplotlib.transforms.offset_copy(ax1.transData, fig=fig1, x=0, y=2.5, units="points")
        bbox = {
            "boxstyle": "round",
            "facecolor": "white",
            "linewidth": 0.25,
            "edgecolor": "black",
            "linestyle": "solid",
            "pad": 0.1,
        }
        ax1.annotate(p_text, (px, py), (0, 4), textcoords="offset points", transform=p_transform, fontsize=6, color="red",
                     horizontalalignment="center", verticalalignment="bottom", bbox=bbox)

    stat_label_xpos = 0.84
    ax1.axhline(y=dbhy_average, xmax=stat_label_xpos, linewidth=0.4, color="red", linestyle="dotted")
    ax1.axhline(y=dbhy_average+2*dbhy_stddev, xmax=stat_label_xpos, linewidth=0.4, color="red", linestyle="dashed")
    ax1.axhline(y=dbhy_average-2*dbhy_stddev, xmax=stat_label_xpos, linewidth=0.4, color="red", linestyle="dashed")
    yline_transform = matplotlib.transforms.blended_transform_factory(ax1.transAxes, ax1.transData)
    bbox = {
        "boxstyle": "round",
        "facecolor": "white",
        "linewidth": 0.25,
        "edgecolor": "red",
        "linestyle": "dashed",
        "pad": 0.1,
    }
    xoffs = 0
    ax1.text(stat_label_xpos+xoffs, dbhy_average, 
             ("%.1f" % (100*dbhy_average,)).replace(".", ",") + r" $(\bar x)$",
             transform=yline_transform, fontsize=6, color="red",
             horizontalalignment="left", verticalalignment="center", bbox=bbox)
    ax1.text(stat_label_xpos+xoffs, dbhy_average+2*dbhy_stddev, 
             ("%.1f" % (100*(dbhy_average+2*dbhy_stddev),)).replace(".", ",") + r" $(+2\sigma)$", 
             transform=yline_transform, fontsize=6, color="red",
             horizontalalignment="left", verticalalignment="center", bbox=bbox)
    ax1.text(stat_label_xpos+xoffs, dbhy_average-2*dbhy_stddev, 
             ("%.1f" % (100*(dbhy_average-2*dbhy_stddev),)).replace(".", ",") + r" $(-2\sigma)$", 
             transform=yline_transform, fontsize=6, color="red",
             horizontalalignment="left", verticalalignment="center", bbox=bbox)
    filename = "figures/process_control_deaths_by_halfyears"
    save_fig(fig1, filename)
    plt.close(fig1)

def get_life_expectancy_buckets(life_expectancy, lower_offset, upper_offset):
    lower_bound = life_expectancy + lower_offset
    lower_bucket = int(lower_bound)
    lower_fraction = 1.0 - (lower_bound - lower_bucket)
    assert 0 <= lower_fraction <= 1
    assert lower_fraction != int(lower_fraction)

    upper_bound = life_expectancy + upper_offset
    upper_bucket = int(upper_bound)+1
    upper_fraction = 1.0 - (upper_bucket - upper_bound)
    assert 0 <= upper_fraction <= 1
    assert upper_fraction != int(upper_fraction)

    return lower_bucket, lower_fraction, upper_bucket, upper_fraction

def test_get_life_expectancy_buckets():
    lb, lf, ub, uf = get_life_expectancy_buckets(79.3, 1, 5)
    result = (lb, round(lf, 1), ub, round(uf, 1))
    expected = (80, 0.7, 85, 0.3)
    assert result == expected, "Invalid result: %s, expected %s" % (repr(result), repr(expected))
test_get_life_expectancy_buckets()

def plot_population_normalization(country_both_life_expectancy_fn, country_male_life_expectancy_fn, country_female_life_expectancy_fn, country_population_by_year_and_age, baseline_trend_fn):
    raise NotImplementedError()
    normalization_x_list = []
    normalization_x_date_list = []
    normalization_y_list = []
    normalization_y2_list = []
    baseline_y_list = []
    male_life_expectancy_y_list = []
    female_life_expectancy_y_list = []
    for year in range(1990, 2020):
        population_by_age = country_population_by_year_and_age[year]
        x_date = datetime.date(year, 1, 1)
        x = (x_date - T0_DATE).days
        normalization_x_list.append(x)
        normalization_x_date_list.append(x_date)
        baseline_y_list.append(baseline_trend_fn([x])[0])
        # Example of age buckets with +/- 3 year bounds with 79,3 years life expectancy:
        #                   |------------------------- 79.3 --------------------------|
        # Age bucket:       76      77      78      79   |  80      81      82      83
        # Bucket fraction:  0.7     1       1       1       1       1       1       0.3
        male_life_expectancy = country_male_life_expectancy_fn([x])[0]
        male_life_expectancy_y_list.append(male_life_expectancy)
        female_life_expectancy = country_female_life_expectancy_fn([x])[0]
        female_life_expectancy_y_list.append(female_life_expectancy)
        male_age_lower_bucket, male_age_lower_fraction, male_age_upper_bucket, male_age_upper_fraction = get_life_expectancy_buckets(male_life_expectancy, 5, 15)
        female_age_lower_bucket, female_age_lower_fraction, female_age_upper_bucket, female_age_upper_fraction = get_life_expectancy_buckets(female_life_expectancy, 5, 15)
        male_sum = 0
        for age in range(male_age_lower_bucket, male_age_upper_bucket+1):
            (_, male_population, _) = population_by_age[age]
            if age == male_age_lower_bucket:
                male_sum += male_age_lower_fraction * male_population
            elif age == male_age_upper_bucket:
                male_sum += male_age_upper_fraction * male_population
            else:
                male_sum += male_population
        female_sum = 0
        for age in range(female_age_lower_bucket, female_age_upper_bucket+1):
            (_, _, female_population) = population_by_age[age]
            if age == female_age_lower_bucket:
                female_sum += female_age_lower_fraction * female_population
            elif age == female_age_upper_bucket:
                female_sum += female_age_upper_fraction * female_population
            else:
                female_sum += female_population
        normalization_y_list.append(male_sum + female_sum)

        male_age_bucket = int(male_life_expectancy)
        male_sum2 = 0
        for age, (_, male_population, _) in population_by_age.items():
            if age == male_age_bucket:
                male_sum2 += (1.0 - (male_life_expectancy - male_age_bucket)) * male_population
            elif age > male_age_bucket:
                male_sum2 += male_population
        female_age_bucket = int(female_life_expectancy)
        female_sum2 = 0
        for age, (_, _, female_population) in population_by_age.items():
            if age == female_age_bucket:
                female_sum2 += (1.0 - (female_life_expectancy - female_age_bucket)) * female_population
            elif age > female_age_bucket:
                female_sum2 += female_population
        normalization_y2_list.append(male_sum2 + female_sum2)
    normalization_x = numpy.array(normalization_x_list)
    normalization_x_date = numpy.array(normalization_x_date_list)
    normalization_y = numpy.array(normalization_y_list)
    normalization_y2 = numpy.array(normalization_y2_list)
    male_life_expectancy_y = numpy.array(male_life_expectancy_y_list)
    female_life_expectancy_y = numpy.array(female_life_expectancy_y_list)
    baseline_y = numpy.array(baseline_y_list)

    fig, ax = plt.subplots(1, 1, figsize=(PAPER_WIDTH_IN, 2))
    # plt.ylim(-1200, 3150)
    # ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(500))
    # plt.xlim(min(population_normalization_x_date) - datetime.timedelta(days=160), max(population_normalization_x_date) + datetime.timedelta(days=int(1.3*365)))
    fig.autofmt_xdate(rotation=45, ha="right", which="both")
    plt.grid(True)
    ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
    ax.tick_params(axis="x", pad=0.5, labelsize=7, labelrotation=60)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment('center')
    if True:
        axb = ax.twinx()
        ax.plot(normalization_x_date, normalization_y, color="C1", linewidth=0.5, linestyle="-")
        ax.scatter(normalization_x_date, normalization_y, color="black", s=3.0)
        ax.set_ylim(0, max(normalization_y)*1.05)
        axb.plot(normalization_x_date, baseline_y, color="C2", linewidth=0.5, linestyle="-")
        axb.scatter(normalization_x_date, baseline_y, color="black", s=3.0)
        axb.set_ylim(0, max(baseline_y)*1.05)
    if False:
        ax.plot(normalization_x_date, normalization_y2, color="C1", linewidth=0.5, linestyle="-")
        ax.scatter(normalization_x_date, normalization_y2, color="black", s=3.0)
        ax.set_ylim(0, max(normalization_y2)*1.05)
    if False:
        ax.plot(normalization_x_date, male_life_expectancy_y, color="C1", linewidth=0.5, linestyle="-")
        ax.scatter(normalization_x_date, male_life_expectancy_y, color="black", s=3.0)
        ax.plot(normalization_x_date, female_life_expectancy_y, color="C2", linewidth=0.5, linestyle="-")
        ax.scatter(normalization_x_date, female_life_expectancy_y, color="black", s=3.0)
    fig.subplots_adjust(bottom=0.12, top=0.99, left=0.05, right=0.99)
    save_fig(fig, "figures/population_normalization")
    plt.close(fig)

def print_top_acm_table(all_cause_mortality):
    acm_sorted = list(sorted(all_cause_mortality, key=lambda x: x[1], reverse=True))
    print("Top deaths per week:")
    print("Year\tWeek\tDeaths")
    for begin_date, deaths in acm_sorted[:20]:
        year, week, _ = begin_date.isocalendar()
        print("%d\t%d\t%d" % (year, week, deaths))

def get_estimate_point(target_date, yearly_mortality):
    days_in_year = (datetime.date(target_date.year, 12, 31) - datetime.date(target_date.year, 1, 1)).days + 1
    return target_date, yearly_mortality/(days_in_year/7)

def main():
    with open("Finland/Finland weekly ACM.csv", "rt", encoding="iso-8859-1") as csv_file:
        finland_acm, finland_acm_by_category = parse_finland_acm_csv(csv_file, trim_weeks_from_end=2)
    # source: Tilastokeskus Väestöennuste 2019 and 2021
    # https://pxnet2.stat.fi/PXWeb/pxweb/fi/StatFin/StatFin__vrm__vaenn/statfin_vaenn_pxt_139e.px/table/tableViewLayout1/
    finland_acm_estimate = [#get_estimate_point(datetime.date(2020, 7, 1), 54054),    # VE2019
                            get_estimate_point(datetime.date(2021, 7, 1), 55987),     # VE2021
                            get_estimate_point(datetime.date(2024, 7, 1), 56605),     # VE2021
                            ]

    with open("Finland/Finland population and deaths by month.csv", "rt", encoding="iso-8859-1") as csv_file:
        finland_deaths_and_population_by_month = parse_finland_deaths_and_population_by_month_csv(csv_file, trim_months_from_end=1)
        assert len(finland_deaths_and_population_by_month) > 50
    with open("Finland/Finland deaths by month.csv", "rt", encoding="iso-8859-1") as csv_file:
        finland_deaths_by_month_since_1945 = parse_finland_deaths_by_month_csv(csv_file)
        assert len(finland_deaths_by_month_since_1945) > 50
    #with open("Finland population by year.csv", "rt", encoding="iso-8859-1") as csv_file:
    #    finland_population_by_year = parse_finland_population_by_month_csv(csv_file)
    #    assert len(finland_population_by_year) > 50
    with open("Finland/Finland covid deaths.csv", "rt", encoding="utf-8") as csv_file:
        finland_covid_data = parse_finland_thl_covid_data_csv(csv_file)
        assert len(finland_covid_data) > 50
    with open("Finland/Finland Population by age.csv", "rt", encoding="iso-8859-1") as csv_file:
        finland_population_by_year_and_age = parse_finland_tilastokeskus_population_csv(csv_file)
        assert len(finland_population_by_year_and_age) > 10
        assert len(finland_population_by_year_and_age[2010]) > 50
    with open("Finland/Finland population forecast.csv", "rt", encoding="iso-8859-1") as csv_file:
        finland_population_forecast = parse_finland_population_forecast_csv(csv_file)
        assert len(finland_population_forecast) > 10
    with open("EuroMoMo/Euromomo all countries z-scores 2022-04-20.csv", "rt", encoding="utf-8") as csv_file:
        euromomo_data = parse_euromomo_zscores_csv(csv_file)
        assert len(euromomo_data) > 50
        finland_euromomo_data = [(d, item["Finland"]) for d, item in euromomo_data if item.get("Finland") is not None]
    # Source: Worldometer - Finland Demographics - Life Expectancy in Finland
    # https://www.worldometers.info/demographics/finland-demographics/#life-exp
    finland_life_expectancy = list(zip([1950,   1955,   1960,   1965,   1970,   1975,   1980,   1985,   1990,   1995,   2000,   2005,   2010,   2015,   2020],
                                       [66.4,   68.19,  69.07,  69.72,  70.94,  72.72,  74.33,  74.79,  75.84,  77.14,  78.4,   79.54,  80.7,   81.64,  82.48],  # Both sexes
                                       [63.02,  64.76,  65.44,  65.82,  66.63,  68.26,  70.04,  70.65,  71.94,  73.43,  74.93,  76.12,  77.67,  78.75,  79.82],  # Males
                                       [69.59,  71.43,  72.55,  73.50,  75.15,  77.04,  78.39,  78.75,  79.58,  80.70,  81.74,  82.91,  83.67,  84.52,  85.14])) # Females
    os.makedirs("figures", exist_ok=True)
    os.makedirs("figures/EuroMoMo", exist_ok=True)
    os.makedirs("data_output", exist_ok=True)

    if True:
        target_acm = finland_acm
        auto_limits = False
    else:
        target_acm = finland_acm_by_category["65-"]
        auto_limits = True
        
    print_top_acm_table(target_acm)
    (
        (acm_raw_x, acm_raw_x_date, acm_raw_y),
        (acm_averaged_x, acm_averaged_x_date, acm_averaged_y),
        (baseline_average_x, baseline_average_x_date, baseline_average_y),
        (acm_estimate_x, acm_estimate_x_date, acm_estimate_y),
        (baseline_trend_fn, baseline_fn),
        (excess_mortality_x, excess_mortality_x_date, excess_mortality_y)
    ) = calculate_acm_baseline_method2(target_acm, None)
    (finland_both_life_expectancy_fn, 
     finland_male_life_expectancy_fn, 
     finland_female_life_expectancy_fn) = calculate_life_expectancy_fn(finland_life_expectancy)
    finland_deaths_and_population_by_month_extended = combine_deaths_by_month(finland_deaths_by_month_since_1945, finland_deaths_and_population_by_month, min_year=1990)
    print("Year\tEstimated yearly mortality")
    for year in range(1990, 2025+1):
        print("%d\t%d" % (year, round(get_model_yearly_mortality(baseline_fn, year))))
    plot_population_forecast_vs_model(finland_population_forecast, baseline_fn)
    plot_monthly_deaths_per_100k(finland_deaths_and_population_by_month, finland_covid_data)
    plot_weekly_deaths_per_age_per_1M(finland_population_by_year_and_age, finland_acm_by_category)
    plot_raw_acm(acm_raw_x, acm_raw_x_date, acm_raw_y, auto_limits)
    plot_acm_baseline_trend(acm_raw_x, acm_raw_x_date, acm_raw_y,
                             acm_averaged_x_date, acm_averaged_x_date, acm_averaged_y,
                             baseline_average_x, baseline_average_x_date, baseline_average_y,
                             acm_estimate_x, acm_estimate_x_date, acm_estimate_y,
                             baseline_trend_fn, auto_limits)
    plot_acm_baseline_fn(acm_raw_x, acm_raw_x_date, acm_raw_y,
                         acm_averaged_x_date, acm_averaged_x_date, acm_averaged_y,
                         baseline_average_x, baseline_average_x_date, baseline_average_y,
                         acm_estimate_x, acm_estimate_x_date, acm_estimate_y,
                         baseline_fn, auto_limits)
    plot_acm_baseline_trend_and_fn(acm_raw_x, acm_raw_x_date, acm_raw_y,
                                   acm_averaged_x_date, acm_averaged_x_date, acm_averaged_y,
                                   baseline_average_x, baseline_average_x_date, baseline_average_y,
                                   acm_estimate_x, acm_estimate_x_date, acm_estimate_y,
                                   baseline_trend_fn, baseline_fn, auto_limits)
    plot_combined_baseline_subplots(acm_raw_x, acm_raw_x_date, acm_raw_y,
                                    acm_averaged_x_date, acm_averaged_x_date, acm_averaged_y,
                                    baseline_average_x, baseline_average_x_date, baseline_average_y,
                                    acm_estimate_x, acm_estimate_x_date, acm_estimate_y,
                                    baseline_trend_fn, baseline_fn, auto_limits)
    plot_excess_mortality(excess_mortality_x, excess_mortality_x_date, excess_mortality_y)
    # yearly cumulative mortality from newyear
    plot_yearly_cumulative_mortality(target_acm, baseline_fn, finland_covid_data, start_week=1)
    # yearly cumulative mortality from spring
    plot_yearly_cumulative_mortality(target_acm, baseline_fn, finland_covid_data, start_week=16)
    plot_all_time_cumulative_excess_mortality(target_acm, baseline_fn)
    plot_covid_cases_and_deaths(finland_covid_data)
    plot_euromomo_zscores(finland_euromomo_data,
                          "figures/euromomo_zscores",
                          "figures/euromomo_zscores_cumulative",
                          "figures/euromomo_zscores_combined")
    plot_highlighted_euromomo_zscores(finland_euromomo_data)
    plot_euromomo_vs_model_cumulative(excess_mortality_x, excess_mortality_x_date, excess_mortality_y, finland_euromomo_data)
    plot_euromomo_correlation(excess_mortality_x, excess_mortality_x_date, excess_mortality_y, finland_euromomo_data)
    plot_processcontrol_deaths_by_halfyears(finland_deaths_and_population_by_month_extended)
    #plot_population_normalization(finland_both_life_expectancy_fn, finland_male_life_expectancy_fn, finland_female_life_expectancy_fn, finland_population_by_year_and_age, baseline_trend_fn)

    sys.exit(0)

    all_euromomo_countries = set()
    for _, item in euromomo_data:
        all_euromomo_countries.update(item.keys())
    for country in sorted(all_euromomo_countries):
        country_euromomo_data = [(d, item[country]) for d, item in euromomo_data if item.get(country)]
        plot_euromomo_zscores(country_euromomo_data, 
                              "figures/EuroMoMo/%s zscores" % (country,), 
                              "figures/EuroMoMo/%s zscores cumulative" % (country,),
                              "figures/EuroMoMo/%s zscores combined" % (country,))
    
if __name__ == "__main__":
    init_latex()
    main()

