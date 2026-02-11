from typing import Literal

from pydantic import BaseModel, ConfigDict

ScryfallCardLegality = Literal["legal", "not_legal", "restricted", "banned"]
ScryfallCardColor = Literal["W", "R", "B", "U", "G"] | None


class ScryfallImageUris(BaseModel):
    small: str
    normal: str
    large: str
    png: str
    art_crop: str
    border_crop: str


class ScryfallCardFace(BaseModel):
    name: str
    mana_cost: str
    type_line: str | None = None
    oracle_text: str | None = None
    colors: list[ScryfallCardColor] | ScryfallCardColor | None = None
    image_uris: ScryfallImageUris | None = None


class ScryfallCardPrices(BaseModel):
    usd: str | None = None
    usd_foil: str | None = None
    usd_etched: str | None = None
    eur: str | None = None
    eur_foil: str | None = None
    tix: str | None = None


class ScryfallCardRelatedUris(BaseModel):
    model_config = ConfigDict(extra="allow")

    gatherer: str | None = None
    tcgplayer_infinite_articles: str | None = None
    tcgplayer_infinite_decks: str | None = None
    edhrec: str | None = None


class ScryfallCardPurchaseUris(BaseModel):
    model_config = ConfigDict(extra="allow")

    tcgplayer: str | None = None
    cardmarket: str | None = None
    cardhoarder: str | None = None


class ScryfallCardBase(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)
    object: Literal["card"]
    oracle_id: str
    multiverse_ids: list[int]
    resource_id: str | None = None
    mtgo_id: int | None = None
    mtgo_foil_id: int | None = None
    arena_id: int | None = None
    tcgplayer_id: int | None = None
    cardmarket_id: int | None = None
    name: str
    lang: str
    released_at: str
    uri: str
    scryfall_uri: str
    layout: str
    highres_image: bool
    image_status: str
    image_uris: ScryfallImageUris | None = None
    card_faces: list[ScryfallCardFace] | None = None
    mana_cost: str | None = None
    cmc: int | float
    type_line: str
    oracle_text: str | None = None
    power: str | None = None
    toughness: str | None = None
    colors: list[ScryfallCardColor] = []
    color_identity: list[ScryfallCardColor]
    keywords: list[str]
    produced_mana: list[str] | None = None
    legalities: dict[str, ScryfallCardLegality]
    games: list[str]
    reserved: bool
    game_changer: bool
    foil: bool
    nonfoil: bool
    finishes: list[str]
    oversized: bool
    promo: bool
    reprint: bool
    variation: bool
    set_id: str
    set: str
    set_name: str
    set_type: str
    set_uri: str
    set_search_uri: str
    scryfall_set_uri: str
    rulings_uri: str
    prints_search_uri: str
    collector_number: str
    digital: bool
    rarity: str
    flavor_text: str | None = None
    card_back_id: str | None = None
    artist: str
    artist_ids: list[str] | None = None
    illustration_id: str | None = None
    border_color: str
    frame: str
    frame_effects: list[str] | None = None
    full_art: bool
    textless: bool
    booster: bool
    story_spotlight: bool
    edhrec_rank: int | None = None
    penny_rank: int | None = None
    prices: ScryfallCardPrices
    related_uris: ScryfallCardRelatedUris | None = None
    purchase_uris: ScryfallCardPurchaseUris | None = None


class ScryfallCard(ScryfallCardBase):
    id: str


class ScryfallCardApiResponse(BaseModel):
    object: Literal["list"]
    total_cards: int
    has_more: bool
    next_page: str | None = None
    data: list[ScryfallCard]
