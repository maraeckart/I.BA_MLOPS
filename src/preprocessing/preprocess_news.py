import argparse
from datetime import datetime
from pathlib import Path
import re

import pandas as pd
from src.storage.gcs_client import GCSClient
from src.utils.config import load_yaml_config

REGION_KEYWORDS = {
    "World": ["world", "global", "international", "worldwide"],
    "Europe": ["europe", "european"],
    "European Union": ["european union", "eu", "brussels"],
    "Africa": ["africa", "african"],
    "Asia": ["asia", "asian"],
    "Middle East": ["middle east", "middle eastern"],
    "North America": ["north america", "north american"],
    "South America": ["south america", "south american"],
    "Latin America": ["latin america", "latin american"],
    "Oceania": ["oceania"],
    "Arctic": ["arctic"],
    "Antarctica": ["antarctica", "antarctic"],

    "Albania": ["albania", "albanian"],
    "Andorra": ["andorra", "andorran"],
    "Austria": ["austria", "austrian", "vienna"],
    "Belarus": ["belarus", "belarusian", "minsk"],
    "Belgium": ["belgium", "belgian"],
    "Bosnia and Herzegovina": ["bosnia and herzegovina", "bosnia", "bosnian", "herzegovina"],
    "Bulgaria": ["bulgaria", "bulgarian", "sofia"],
    "Croatia": ["croatia", "croatian", "zagreb"],
    "Cyprus": ["cyprus", "cypriot"],
    "Czech Republic": ["czech republic", "czechia", "czech", "prague"],
    "Denmark": ["denmark", "danish", "copenhagen"],
    "Estonia": ["estonia", "estonian", "tallinn"],
    "Finland": ["finland", "finnish", "helsinki"],
    "France": ["france", "french", "paris"],
    "Germany": ["germany", "german", "berlin"],
    "Greece": ["greece", "greek", "athens"],
    "Hungary": ["hungary", "hungarian", "budapest"],
    "Iceland": ["iceland", "icelandic", "reykjavik"],
    "Ireland": ["ireland", "irish", "dublin"],
    "Italy": ["italy", "italian", "rome"],
    "Kosovo": ["kosovo"],
    "Latvia": ["latvia", "latvian", "riga"],
    "Liechtenstein": ["liechtenstein"],
    "Lithuania": ["lithuania", "lithuanian", "vilnius"],
    "Luxembourg": ["luxembourg"],
    "Malta": ["malta", "maltese"],
    "Moldova": ["moldova", "moldovan"],
    "Monaco": ["monaco"],
    "Montenegro": ["montenegro"],
    "Netherlands": ["netherlands", "dutch", "amsterdam"],
    "North Macedonia": ["north macedonia", "macedonia", "macedonian"],
    "Norway": ["norway", "norwegian", "oslo"],
    "Poland": ["poland", "polish", "warsaw"],
    "Portugal": ["portugal", "portuguese", "lisbon"],
    "Romania": ["romania", "romanian", "bucharest"],
    "Russia": ["russia", "russian", "moscow"],
    "San Marino": ["san marino"],
    "Serbia": ["serbia", "serbian", "belgrade"],
    "Slovakia": ["slovakia", "slovak", "bratislava"],
    "Slovenia": ["slovenia", "slovenian", "ljubljana"],
    "Spain": ["spain", "spanish", "madrid"],
    "Sweden": ["sweden", "swedish", "stockholm"],
    "Switzerland": ["switzerland", "swiss", "zurich", "geneva", "bern"],
    "Turkey": ["turkey", "turkish", "ankara", "istanbul"],
    "Ukraine": ["ukraine", "ukrainian", "kyiv", "kiev"],
    "United Kingdom": [
        "united kingdom",
        "britain",
        "british",
        "england",
        "scotland",
        "wales",
        "northern ireland",
        "london",
    ],
    "Vatican City": ["vatican", "vatican city"],

    "United States": [
        "united states",
        "us",
        "u.s.",
        "usa",
        "america",
        "american",
        "washington",
        "new york",
        "california",
        "texas",
        "florida",
    ],
    "Canada": ["canada", "canadian", "ottawa", "toronto", "vancouver", "montreal"],
    "Mexico": ["mexico", "mexican", "mexico city"],

    "Belize": ["belize"],
    "Costa Rica": ["costa rica", "costa rican"],
    "El Salvador": ["el salvador", "salvadoran"],
    "Guatemala": ["guatemala", "guatemalan"],
    "Honduras": ["honduras", "honduran"],
    "Nicaragua": ["nicaragua", "nicaraguan"],
    "Panama": ["panama", "panamanian"],
    "Cuba": ["cuba", "cuban", "havana"],
    "Haiti": ["haiti", "haitian"],
    "Dominican Republic": ["dominican republic", "dominican"],
    "Jamaica": ["jamaica", "jamaican"],
    "Bahamas": ["bahamas"],
    "Barbados": ["barbados"],
    "Trinidad and Tobago": ["trinidad and tobago", "trinidad", "tobago"],

    "Argentina": ["argentina", "argentinian", "argentine", "buenos aires"],
    "Bolivia": ["bolivia", "bolivian"],
    "Brazil": ["brazil", "brazilian", "brasilia", "rio de janeiro", "sao paulo"],
    "Chile": ["chile", "chilean", "santiago"],
    "Colombia": ["colombia", "colombian", "bogota"],
    "Ecuador": ["ecuador", "ecuadorian", "quito"],
    "Guyana": ["guyana"],
    "Paraguay": ["paraguay", "paraguayan"],
    "Peru": ["peru", "peruvian", "lima"],
    "Suriname": ["suriname"],
    "Uruguay": ["uruguay", "uruguayan"],
    "Venezuela": ["venezuela", "venezuelan", "caracas"],

    "Iran": ["iran", "iranian", "tehran"],
    "Iraq": ["iraq", "iraqi", "baghdad"],
    "Israel": ["israel", "israeli", "jerusalem", "tel aviv"],
    "Palestine": ["palestine", "palestinian", "gaza", "west bank"],
    "Lebanon": ["lebanon", "lebanese", "beirut"],
    "Syria": ["syria", "syrian", "damascus"],
    "Jordan": ["jordan", "jordanian", "amman"],
    "Saudi Arabia": ["saudi arabia", "saudi", "riyadh"],
    "United Arab Emirates": ["united arab emirates", "uae", "dubai", "abu dhabi"],
    "Qatar": ["qatar", "qatari", "doha"],
    "Kuwait": ["kuwait", "kuwaiti"],
    "Bahrain": ["bahrain", "bahraini"],
    "Oman": ["oman", "omani"],
    "Yemen": ["yemen", "yemeni"],

    "Afghanistan": ["afghanistan", "afghan", "kabul"],
    "Armenia": ["armenia", "armenian", "yerevan"],
    "Azerbaijan": ["azerbaijan", "azerbaijani", "baku"],
    "Bangladesh": ["bangladesh", "bangladeshi", "dhaka"],
    "Bhutan": ["bhutan", "bhutanese"],
    "Cambodia": ["cambodia", "cambodian", "phnom penh"],
    "China": ["china", "chinese", "beijing", "shanghai", "hong kong"],
    "Georgia": ["georgia", "georgian", "tbilisi"],
    "India": ["india", "indian", "delhi", "new delhi", "mumbai"],
    "Indonesia": ["indonesia", "indonesian", "jakarta", "bali"],
    "Japan": ["japan", "japanese", "tokyo"],
    "Kazakhstan": ["kazakhstan", "kazakh", "astana"],
    "Kyrgyzstan": ["kyrgyzstan", "kyrgyz"],
    "Laos": ["laos", "laotian"],
    "Malaysia": ["malaysia", "malaysian", "kuala lumpur"],
    "Maldives": ["maldives"],
    "Mongolia": ["mongolia", "mongolian"],
    "Myanmar": ["myanmar", "burma", "burmese"],
    "Nepal": ["nepal", "nepalese", "kathmandu"],
    "North Korea": ["north korea", "north korean", "pyongyang"],
    "Pakistan": ["pakistan", "pakistani", "islamabad", "karachi"],
    "Philippines": ["philippines", "philippine", "filipino", "manila"],
    "Singapore": ["singapore", "singaporean"],
    "South Korea": ["south korea", "south korean", "seoul"],
    "Sri Lanka": ["sri lanka", "sri lankan", "colombo"],
    "Taiwan": ["taiwan", "taiwanese", "taipei"],
    "Tajikistan": ["tajikistan", "tajik"],
    "Thailand": ["thailand", "thai", "bangkok"],
    "Turkmenistan": ["turkmenistan", "turkmen"],
    "Uzbekistan": ["uzbekistan", "uzbek"],
    "Vietnam": ["vietnam", "vietnamese", "hanoi", "ho chi minh"],

    "Algeria": ["algeria", "algerian", "algiers"],
    "Angola": ["angola", "angolan"],
    "Botswana": ["botswana"],
    "Cameroon": ["cameroon", "cameroonian"],
    "Democratic Republic of the Congo": [
        "democratic republic of the congo",
        "dr congo",
        "drc",
        "congolese",
        "kinshasa",
    ],
    "Republic of the Congo": ["republic of the congo", "congo-brazzaville", "brazzaville"],
    "Egypt": ["egypt", "egyptian", "cairo"],
    "Ethiopia": ["ethiopia", "ethiopian", "addis ababa"],
    "Ghana": ["ghana", "ghanaian", "accra"],
    "Ivory Coast": ["ivory coast", "cote d'ivoire", "côte d’ivoire"],
    "Kenya": ["kenya", "kenyan", "nairobi"],
    "Libya": ["libya", "libyan", "tripoli"],
    "Madagascar": ["madagascar", "malagasy"],
    "Mali": ["mali", "malian"],
    "Morocco": ["morocco", "moroccan", "rabat", "casablanca"],
    "Mozambique": ["mozambique", "mozambican"],
    "Namibia": ["namibia", "namibian"],
    "Nigeria": ["nigeria", "nigerian", "abuja", "lagos"],
    "Rwanda": ["rwanda", "rwandan", "kigali"],
    "Senegal": ["senegal", "senegalese", "dakar"],
    "Somalia": ["somalia", "somali", "mogadishu"],
    "South Africa": ["south africa", "south african", "cape town", "johannesburg", "pretoria"],
    "Sudan": ["sudan", "sudanese", "khartoum"],
    "Tanzania": ["tanzania", "tanzanian", "dar es salaam"],
    "Tunisia": ["tunisia", "tunisian", "tunis"],
    "Uganda": ["uganda", "ugandan", "kampala"],
    "Zambia": ["zambia", "zambian"],
    "Zimbabwe": ["zimbabwe", "zimbabwean"],

    "Australia": ["australia", "australian", "canberra", "sydney", "melbourne"],
    "New Zealand": ["new zealand", "new zealander", "auckland", "wellington"],
    "Fiji": ["fiji", "fijian"],
    "Papua New Guinea": ["papua new guinea"],
    "Samoa": ["samoa", "samoan"],
    "Tonga": ["tonga", "tongan"],
}


def clean_text(text: str | None) -> str:
    if text is None:
        return ""

    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9äöüÄÖÜßéèàçÉÈÀÇ\s]", "", text)
    text = text.strip()
    return text


def clean_news_data(news_df: pd.DataFrame) -> pd.DataFrame:
    news_df = news_df.copy()

    news_df = news_df.dropna(subset=["headline"])
    news_df = news_df.drop_duplicates(subset=["headline", "url"])

    news_df["description"] = news_df["description"].fillna("")

    news_df["headline_clean"] = news_df["headline"].apply(clean_text)
    news_df["description_clean"] = news_df["description"].apply(clean_text)
    return news_df


def add_text_features(news_df: pd.DataFrame) -> pd.DataFrame:
    news_df = news_df.copy()

    news_df["headline_length"] = news_df["headline_clean"].str.len()
    news_df["headline_word_count"] = news_df["headline_clean"].str.split().str.len()

    news_df["description_length"] = news_df["description_clean"].str.len()
    news_df["description_word_count"] = news_df["description_clean"].str.split().str.len()

    return news_df


def add_time_features(news_df: pd.DataFrame) -> pd.DataFrame:
    news_df = news_df.copy()

    news_df["published_at_parsed"] = pd.to_datetime(
        news_df["published_at"],
        errors="coerce",
        utc=True,
    )

    news_df["publication_hour"] = news_df["published_at_parsed"].dt.hour
    news_df["publication_day_of_week"] = news_df["published_at_parsed"].dt.day_name()
    return news_df

def combine_text_for_region(headline: str | None, description: str | None) -> str:
    text_parts = []

    if isinstance(headline, str):
        text_parts.append(headline)

    if isinstance(description, str):
        text_parts.append(description)

    combined_text = " ".join(text_parts)
    combined_text = combined_text.lower()

    return combined_text


def contains_keyword(text: str, keyword: str) -> bool:
    text_with_spaces = f" {text} "
    keyword_with_spaces = f" {keyword} "

    return keyword_with_spaces in text_with_spaces


def extract_region(headline: str | None, description: str | None) -> str:
    text = combine_text_for_region(headline, description)

    for region, keywords in REGION_KEYWORDS.items():
        for keyword in keywords:
            if contains_keyword(text, keyword):
                return region

    return "Unknown"


def add_region_feature(news_df: pd.DataFrame) -> pd.DataFrame:
    news_df = news_df.copy()

    regions = []

    for row in news_df.itertuples(index=False):
        region = extract_region(
            headline=getattr(row, "headline", None),
            description=getattr(row, "description", None),
        )
        regions.append(region)

    news_df["region"] = regions

    return news_df


def add_features(news_df: pd.DataFrame) -> pd.DataFrame:
    news_df = add_text_features(news_df)
    news_df = add_time_features(news_df)
    news_df = add_region_feature(news_df)
    return news_df


def get_raw_file_for_date(run_date: str, raw_dir: str = "data/raw/rss/live") -> Path:
    raw_file = Path(raw_dir) / f"raw_guardian_news_{run_date}.csv"

    if not raw_file.exists():
        raise FileNotFoundError(f"Raw file not found: {raw_file}")

    return raw_file


def preprocess_news(input_path: str | Path) -> pd.DataFrame:
    news_df = pd.read_csv(input_path)

    news_df = clean_news_data(news_df)
    news_df = add_features(news_df)

    return news_df


def save_processed_news(
    news_df: pd.DataFrame,
    output_path: str | Path,
) -> str:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    news_df.to_csv(output_path, index=False, encoding="utf-8")

    return str(output_path)


def get_default_output_path(run_date: str) -> Path:
    output_path = Path("data/processed/live") / f"processed_guardian_news_{run_date}.csv"

    return output_path



def get_api_output_path(input_path: str | Path) -> Path:
    input_path = Path(input_path)
    file_name = input_path.name

    processed_file_name = file_name.replace("raw_", "processed_")
    output_path = Path("data/processed/api") / processed_file_name

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-date",
        required=False,
        default=None,
        help="Pipeline run date in YYYY-MM-DD format.",
    )

    parser.add_argument(
        "--input-path",
        required=False,
        default=None,
        help="Optional raw input CSV path. Use this for API backfill files.",
    )

    parser.add_argument(
        "--output-path",
        required=False,
        default=None,
        help="Optional processed output CSV path.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.input_path:
        raw_file = Path(args.input_path)

        if not raw_file.exists():
            raise FileNotFoundError(f"Input file not found: {raw_file}")

        if args.output_path:
            output_path = Path(args.output_path)
        else:
            output_path = get_api_output_path(raw_file)

    else:
        if args.run_date:
            run_date = args.run_date
        else:
            run_date = datetime.now().strftime("%Y-%m-%d")

        raw_file = get_raw_file_for_date(run_date)
        output_path = get_default_output_path(run_date)

    print(f"Using raw file: {raw_file}")

    processed_df = preprocess_news(raw_file)

    print(f"Processed articles: {len(processed_df)}")

    saved_path = save_processed_news(
        news_df=processed_df,
        output_path=output_path,
    )
    print(f"Saved processed news to: {saved_path}")


if __name__ == "__main__":
    main()