"""Module to generate the commands dataset."""
import logging

from nltk import grammar, parse
from nltk.parse.generate import generate
import pandas as pd

_LOGGER = logging.getLogger(__name__)


def turn_onoff_response(area, domain, device_class):
    """Return response for a HassTurnOn or HassTurnOff command."""
    if device_class == "Garage":
        return "cover_garage"

    if area != "none":
        if domain == "light":
            return "lights_area"
        if domain == "fan":
            return "fans_area"
        if domain == "cover" and device_class == "none":
            return "cover_area"
        if domain == "cover":
            return "cover_device_class_area"
    else:
        if domain == "cover" and device_class == "none":
            return "cover"
        if domain == "cover":
            return "cover_device_class"

    return "default"


def generate_artificial_dataset(dataset_file_path):
    """Generate smart home commands from a context-free grammar."""
    g = """
    ## NLTK-style feature-based CFG

    % start S

    S -> Intent
    Intent -> HassTurnOn | HassTurnOff | HassGetState | HassLightSet | HassClimateGetTemperature | HassClimateSetTemperature
    HassTurnOn -> Light_TurnOn | Fan_TurnOn | Cover_Open | Entity_TurnOn
    HassTurnOff -> Light_TurnOff | Fan_TurnOff | Cover_Close | Entity_TurnOff
    HassGetState -> Cover_Get | Entity_Get
    HassLightSet -> Light_SetBrightness | Light_SetColor
    HassClimateGetTemperature -> Climate_Get
    HassClimateSetTemperature -> Climate_Set

    Percentage -> NumPer | NumPer 'percento'
    NumPer -> 'zero' | 'cinquanta' | 'cento'

    OnOffDomain[NUM=?n, GEN=?g, ART=?a] -> Fan[NUM=?n, GEN=?g, ART=?a]
    OnOffDomain[NUM=?n, GEN=f] -> Light[NUM=?n, GEN=f]
    OnOffState[NUM=?n, GEN=?g] -> OnState[NUM=?n, GEN=?g] | OffState[NUM=?n, GEN=?g]
    OpenCloseState[NUM=?n, GEN=?g] -> OpenState[NUM=?n, GEN=?g] | CloseState[NUM=?n, GEN=?g]
    OnState[NUM=sg, GEN=m] -> 'acceso'
    OnState[NUM=sg, GEN=f] -> 'accesa'
    OffState[NUM=sg, GEN=m] -> 'spento'
    OffState[NUM=sg, GEN=f] -> 'spenta'
    OpenState[NUM=sg, GEN=m] -> 'aperto'
    OpenState[NUM=sg, GEN=f] -> 'aperta'
    CloseState[NUM=sg, GEN=m] -> 'chiuso'
    CloseState[NUM=sg, GEN=f] -> 'chiusa'
    OnState[NUM=pl, GEN=m] -> 'accesi'
    OnState[NUM=pl, GEN=f] -> 'accese'
    OffState[NUM=pl, GEN=m] -> 'spenti'
    OffState[NUM=pl, GEN=f] -> 'spente'
    OpenState[NUM=pl, GEN=m] -> 'aperti'
    OpenState[NUM=pl, GEN=f] -> 'aperte'
    CloseState[NUM=pl, GEN=m] -> 'chiusi'
    CloseState[NUM=pl, GEN=f] -> 'chiuse'

    Light[NUM=sg, GEN=f] -> 'luce' | 'lampada'
    Light[NUM=pl, GEN=f] -> 'luci' | 'lampade'
    Fan[NUM=sg, GEN=f] -> 'ventola' | 'ventilazione'
    Fan[NUM=sg, GEN=m, ART=il] -> 'ventilatore' | 'climatizzatore' | 'condizionatore'
    Fan[NUM=pl, GEN=f] -> 'ventole'
    Fan[NUM=pl, GEN=m, ART=il] -> 'ventilatori' | 'climatizzatori' | 'condizionatori'
    Awning[NUM=sg, GEN=f] -> 'tenda da sole'
    Awning[NUM=pl, GEN=f] -> 'tende da sole'
    Blind[NUM=sg, GEN=f] -> 'persiana'
    Blind[NUM=pl, GEN=f] -> 'persiane'
    Curtain[NUM=sg, GEN=f] -> 'tenda'
    Curtain[NUM=pl, GEN=f] -> 'tende'
    Door[NUM=sg, GEN=f] -> 'porta'
    Door[NUM=pl, GEN=f] -> 'porte'
    Garage[NUM=sg, GEN=f] ->  'serranda' | 'saracinesca'
    Garage[NUM=sg, GEN=m, ART=il] -> 'garage'
    Garage[NUM=pl, GEN=f] ->  'serrande' | 'saracinesche'
    Gate[NUM=sg, GEN=m, ART=il] -> 'cancello'
    Gate[NUM=pl, GEN=m, ART=il] -> 'cancelli'
    Shade[NUM=sg, GEN=f] -> 'veneziana'
    Shade[NUM=pl, GEN=f] -> 'veneziane'
    Shutter[NUM=sg, GEN=f] -> 'tapparella'
    Shutter[NUM=pl, GEN=f] -> 'tapparelle'
    Window[NUM=sg, GEN=f] -> 'finestra'
    Window[NUM=pl, GEN=f] -> 'finestre'

    TurnOn -> 'accendi' | CanYouDo 'accendere' | 'attiva' | CanYouDo 'attivare'
    TurnOff -> 'spegni' | CanYouDo 'spegnere' | 'disattiva' | CanYouDo 'disattivare'
    Open -> 'apri' | CanYouDo 'aprire'
    Close -> 'chiudi' | CanYouDo 'chiudere'
    Set -> 'imposta' | CanYouDo 'impostare'
    Change ->  'cambia' | CanYouDo 'cambiare'
    Area -> 'custom_area'
    Name -> 'custom_name'

    CanYouDo -> 'potresti' | 'puoi'
    CanYouTell -> 'dimmi' | 'puoi dirmi'
    A[NUM=sg, GEN=m, ART=il] -> 'un'
    A[NUM=sg, GEN=m, ART=lo] -> 'uno'
    A[NUM=sg, GEN=f] -> 'una'
    A[NUM=pl, GEN=m, ART=il] -> 'dei'
    A[NUM=pl, GEN=m, ART=lo] -> 'degli'
    A[NUM=pl, GEN=f] -> 'delle'
    The[NUM=sg, GEN=m, ART=il] -> 'il'
    The[NUM=sg, GEN=m, ART=lo] -> 'lo'
    The[NUM=sg, GEN=f] -> 'la'
    The[NUM=pl, GEN=m, ART=il] -> 'i'
    The[NUM=pl, GEN=m, ART=lo] -> 'gli'
    The[NUM=pl, GEN=f] -> 'le'
    In[NUM=sg, GEN=m, ART=il] -> 'in' | 'nel'
    In[NUM=sg, GEN=m, ART=lo] -> 'nello'
    In[NUM=sg, GEN=f] -> 'nella'
    In[NUM=pl, GEN=m, ART=il] -> 'nei'
    In[NUM=pl, GEN=m, ART=il] -> 'negli'
    In[NUM=pl, GEN=f] -> 'nelle'
    Of[NUM=sg, GEN=m, ART=il] -> 'del'
    Of[NUM=sg, GEN=m, ART=lo] -> 'dello'
    Of[NUM=sg, GEN=f] ->  'della'
    Of[NUM=pl, GEN=m, ART=il] -> 'dei'
    Of[NUM=pl, GEN=m, ART=lo] -> 'degli'
    Of[NUM=pl, GEN=f] -> 'delle'
    Onto -> 'a' | 'su'
    Into -> 'in'
    To[NUM=sg, GEN=m, ART=il] -> 'al'
    To[NUM=sg, GEN=m, ART=il] -> 'allo'
    To[NUM=sg, GEN=f] ->  'alla'
    To[NUM=pl, GEN=m, ART=il] -> 'ai'
    To[NUM=pl, GEN=m, ART=lo] -> 'agli'
    To[NUM=pl, GEN=f] -> 'alle'
    Is[NUM=sg] -> 'è'
    Is[NUM=pl] -> 'sono'
    ThereIs[NUM=sg] -> "c'è"
    ThereIs[NUM=pl] -> "ci sono"
    WhatIs -> 'qual è' | "com'è" | "quant'è"
    What[NUM=sg] -> 'quale'
    What[NUM=pl] -> 'quali'
    HowMany[GEN=m] -> 'quanti'
    HowMany[GEN=f] -> 'quante'
    TellIf -> CanYouTell 'se'
    Every[GEN=m] -> 'tutti'
    Every[GEN=f] -> 'tutte'
    Where -> In[NUM=sg, GEN=?g, ART=?a] Area[GEN=?g, ART=?a]
    WhereOf -> Where | Of[NUM=sg, GEN=?g, ART=?a] Area[GEN=?g, ART=?a]

    Entity_TurnOn -> TurnOn EntitySubject
    Entity_TurnOff -> TurnOff EntitySubject

    Entity_Get -> One | One_YesNo | Any | All | Which | How_Many

    One -> 'Qual è lo stato' Of[NUM=?n, GEN=?g, ART=?a] Name

    One_YesNo -> TellIf OneYesNoQuestion | OneYesNoQuestion
    OneYesNoQuestion -> EntitySubject[NUM=?n, GEN=?n, ART=?a] Is[NUM=?n] OnOffState[NUM=?n, GEN=?g]
    OneYesNoQuestion -> Is[NUM=?n] OnOffState[NUM=?n, GEN=?g] EntitySubject[NUM=?n, GEN=?n, ART=?a]
    OneYesNoQuestion -> EntitySubject[NUM=?n, GEN=?n, ART=?a] Is[NUM=?n] OnOffState[NUM=?n, GEN=?g] Where
    OneYesNoQuestion -> Is[NUM=?n] OnOffState[NUM=?n, GEN=?g] EntitySubject[NUM=?n, GEN=?n, ART=?a] Where
    OneYesNoQuestion -> EntitySubject[NUM=?n, GEN=?n, ART=?a] WhereOf Is[NUM=?n] OnOffState[NUM=?n, GEN=?g]

    Any -> TellIf AnyQuestion | AnyQuestion
    AnyQuestion[NUM=?n, GEN=?g, ART=?a] -> ThereIs[NUM=?n] A[NUM=?n, GEN=?g, ART=?a] OnOffDomain[NUM=?n, GEN=?g, ART=?a] OnOffState[NUM=?n, GEN=?g]
    AnyQuestion[NUM=?n, GEN=?g, ART=?a] -> ThereIs[NUM=?n] A[NUM=?n, GEN=?g, ART=?a] OnOffDomain[NUM=?n, GEN=?g, ART=?a] WhereOf OnOffState[NUM=?n, GEN=?g]
    AnyQuestion[NUM=?n, GEN=?g, ART=?a] -> ThereIs[NUM=?n] A[NUM=?n, GEN=?g, ART=?a] OnOffDomain[NUM=?n, GEN=?g, ART=?a] OnOffState[NUM=?n, GEN=?g] Where

    All -> TellIf AllQuestion | AllQuestion
    AllQuestion[NUM=pl, GEN=?g, ART=?a] -> Every[GEN=?g] The[NUM=pl, GEN=?g, ART=?a] OnOffDomain[NUM=pl, GEN=?g, ART=?a] Is[NUM=pl] OnOffState[NUM=pl, GEN=?g]
    AllQuestion[NUM=pl, GEN=?g, ART=?a] -> Every[GEN=?g] The[NUM=pl, GEN=?g, ART=?a] OnOffDomain[NUM=pl, GEN=?g, ART=?a] WhereOf Is[NUM=pl] OnOffState[NUM=pl, GEN=?g]
    AllQuestion[NUM=pl, GEN=?g, ART=?a] -> Every[GEN=?g] The[NUM=pl, GEN=?g, ART=?a] OnOffDomain[NUM=pl, GEN=?g, ART=?a] Is[NUM=pl] OnOffState[NUM=pl, GEN=?g] Where

    Which -> CanYouTell WhichQuestion | WhichQuestion
    WhichQuestion[NUM=?n, GEN=?g, ART=?a] -> What[NUM=?n] OnOffDomain[NUM=?n, GEN=?g, ART=?a] Is[NUM=?n] OnOffState[NUM=?n, GEN=?g]
    WhichQuestion[NUM=?n, GEN=?g, ART=?a] -> What[NUM=?n] OnOffDomain[NUM=?n, GEN=?g, ART=?a] WhereOf Is[NUM=?n] OnOffState[NUM=?n, GEN=?g]
    WhichQuestion[NUM=?n, GEN=?g, ART=?a] -> What[NUM=?n] OnOffDomain[NUM=?n, GEN=?g, ART=?a] Is[NUM=?n] OnOffState[NUM=?n, GEN=?g] Where

    How_Many -> CanYouTell HowManyQuestion | HowManyQuestion
    HowManyQuestion[NUM=?n, GEN=?g, ART=?a] -> HowMany[GEN=?g] OnOffDomain[NUM=pl, GEN=?g, ART=?a] WhereOf Is[NUM=pl] OnOffState[NUM=pl, GEN=?g]
    HowManyQuestion[NUM=?n, GEN=?g, ART=?a] -> HowMany[GEN=?g] OnOffDomain[NUM=pl, GEN=?g, ART=?a] WhereOf Is[NUM=pl] OnOffState[NUM=pl, GEN=?g]
    HowManyQuestion[NUM=?n, GEN=?g, ART=?a] -> HowMany[GEN=?g] OnOffDomain[NUM=pl, GEN=?g, ART=?a] Is[NUM=pl] OnOffState[NUM=pl, GEN=?g] Where

    EntitySubject[NUM=?n, GEN=?g, ART=?a] -> Name | The[NUM=sg, GEN=?g, ART=?a] Name[NUM=sg, GEN=?g, ART=?a]

    Cover_Get -> Cover_One_YesNo | Cover_Any | Cover_All | Cover_Which | Cover_How_Many

    Cover_One_YesNo -> TellIf Cover_OneYesNoQuestion | Cover_OneYesNoQuestion
    Cover_OneYesNoQuestion -> CoverSubject[NUM=?n, GEN=?g, ART=?a] Is[NUM=?n] OpenCloseState[NUM=?n, GEN=?g]
    Cover_OneYesNoQuestion -> Is[NUM=?n] OpenCloseState[NUM=?n, GEN=?g] CoverSubject[NUM=?n, GEN=?g, ART=?a]
    Cover_OneYesNoQuestion -> InteriorCoverSubject[NUM=?n, GEN=?g, ART=?a] Is[NUM=?n] OpenCloseState[NUM=?n, GEN=?g] Where
    Cover_OneYesNoQuestion -> Is[NUM=?n] OpenCloseState[NUM=?n, GEN=?g] InteriorCoverSubject[NUM=?n, GEN=?g, ART=?a] Where
    Cover_OneYesNoQuestion -> InteriorCoverSubject[NUM=?n, GEN=?g, ART=?a] WhereOf Is[NUM=?n] OpenCloseState[NUM=?n, GEN=?g]

    Cover_Any -> TellIf Cover_AnyQuestion | Cover_AnyQuestion
    Cover_AnyQuestion[NUM=?n, GEN=?g, ART=?a] -> ThereIs[NUM=?n] A[NUM=?n, GEN=?g, ART=?a] Cover[NUM=?n, GEN=?g, ART=?a] OpenCloseState[NUM=?n, GEN=?g]
    Cover_AnyQuestion[NUM=?n, GEN=?g, ART=?a] -> ThereIs[NUM=?n] A[NUM=?n, GEN=?g, ART=?a] InteriorCover[NUM=?n, GEN=?g, ART=?a] WhereOf OpenCloseState[NUM=?n, GEN=?g]
    Cover_AnyQuestion[NUM=?n, GEN=?g, ART=?a] -> ThereIs[NUM=?n] A[NUM=?n, GEN=?g, ART=?a] InteriorCover[NUM=?n, GEN=?g, ART=?a] OpenCloseState[NUM=?n, GEN=?g] Where

    Cover_All -> TellIf Cover_AllQuestion | Cover_AllQuestion
    Cover_AllQuestion[NUM=pl, GEN=?g, ART=?a] -> Every[GEN=?g] The[NUM=pl, GEN=?g, ART=?a] Cover[NUM=pl, GEN=?g, ART=?a] Is[NUM=pl] OpenCloseState[NUM=pl, GEN=?g]
    Cover_AllQuestion[NUM=pl, GEN=?g, ART=?a] -> Every[GEN=?g] The[NUM=pl, GEN=?g, ART=?a] InteriorCover[NUM=pl, GEN=?g, ART=?a] WhereOf Is[NUM=pl] OpenCloseState[NUM=pl, GEN=?g]
    Cover_AllQuestion[NUM=pl, GEN=?g, ART=?a] -> Every[GEN=?g] The[NUM=pl, GEN=?g, ART=?a] InteriorCover[NUM=pl, GEN=?g, ART=?a] Is[NUM=pl] OpenCloseState[NUM=pl, GEN=?g] Where

    Cover_Which -> CanYouTell Cover_WhichQuestion | Cover_WhichQuestion
    Cover_WhichQuestion[NUM=?n, GEN=?g, ART=?a] -> What[NUM=?n] Cover[NUM=?n, GEN=?g, ART=?a] Is[NUM=?n] OpenCloseState[NUM=?n, GEN=?g]
    Cover_WhichQuestion[NUM=?n, GEN=?g, ART=?a] -> What[NUM=?n] InteriorCover[NUM=?n, GEN=?g, ART=?a] WhereOf Is[NUM=?n] OpenCloseState[NUM=?n, GEN=?g]
    Cover_WhichQuestion[NUM=?n, GEN=?g, ART=?a] -> What[NUM=?n] InteriorCover[NUM=?n, GEN=?g, ART=?a] Is[NUM=?n] OpenCloseState[NUM=?n, GEN=?g] Where

    Cover_How_Many -> CanYouTell Cover_HowManyQuestion | Cover_HowManyQuestion
    Cover_HowManyQuestion[NUM=?n, GEN=?g, ART=?a] -> HowMany[GEN=?g] Cover[NUM=pl, GEN=?g, ART=?a] Is[NUM=pl] OpenCloseState[NUM=pl, GEN=?g]
    Cover_HowManyQuestion[NUM=?n, GEN=?g, ART=?a] -> HowMany[GEN=?g] InteriorCover[NUM=pl, GEN=?g, ART=?a] WhereOf Is[NUM=pl] OpenCloseState[NUM=pl, GEN=?g]
    Cover_HowManyQuestion[NUM=?n, GEN=?g, ART=?a] -> HowMany[GEN=?g] InteriorCover[NUM=pl, GEN=?g, ART=?a] Is[NUM=pl] OpenCloseState[NUM=pl, GEN=?g] Where

    CoverSubject[NUM=?n, GEN=?g, ART=?a] -> The[NUM=?n, GEN=?g, ART=?a] Cover[NUM=?n, GEN=?g, ART=?a] | Cover[NUM=?n, GEN=?g, ART=?a]
    InteriorCoverSubject[NUM=?n, GEN=?g, ART=?a] -> The[NUM=?n, GEN=?g, ART=?a] InteriorCover[NUM=?n, GEN=?g, ART=?a]

    Light_TurnOn -> TurnOn LightSubject | TurnOn LightSubject WhereOf
    Light_TurnOff -> TurnOff LightSubject | TurnOff LightSubject WhereOf

    Light_SetBrightness -> Set Brightness Onto Percentage | Set Onto Percentage Brightness
    Light_SetBrightness -> Set Brightness WhereOf Onto Percentage | Set Onto Percentage Brightness WhereOf
    Light_SetBrightness -> Change Brightness Into Percentage | Change Into Percentage Brightness
    Light_SetBrightness -> Change Brightness WhereOf Into Percentage | Change Into Percentage Brightness WhereOf
    Brightness ->  The[NUM=sg, GEN=f] 'luminosità' Of[NUM=?n, GEN=f] Light[NUM=?n, GEN=f]

    Light_SetColor ->  Set Color WhereOf Onto ColorValue | Set Onto ColorValue Color WhereOf
    Light_SetColor ->  Change Color WhereOf Into ColorValue | Change Into ColorValue Color WhereOf
    Color ->  The[NUM=sg, GEN=m, ART=il] 'colore' Of[NUM=?n, GEN=f] Light[NUM=?n, GEN=f]
    ColorValue -> 'bianco' | 'blu' | 'giallo'

    LightSubject[NUM=?n, GEN=f] -> The[NUM=?n, GEN=f] Light[NUM=?n] | Light[NUM=?n]
    LightSubject[NUM=pl, GEN=f] -> Every[GEN=f] The[NUM=pl, GEN=f] Light[NUM=pl]

    Fan_TurnOn -> TurnOn FanSubject | TurnOn FanSubject WhereOf
    Fan_TurnOff -> TurnOff FanSubject | TurnOff FanSubject WhereOf

    FanSubject[NUM=?n, GEN=?g, ART=?a] -> The[NUM=?n, GEN=?g, ART=?a] Fan[NUM=?n, GEN=?g, ART=?a] | Fan[NUM=?n, GEN=?g, ART=?a]
    FanSubject[NUM=pl, GEN=?g, ART=?a] -> Every[GEN=?g] The[NUM=pl, GEN=?g, ART=?a] Fan[NUM=pl, GEN=?g, ART=?a]

    Cover_Open -> Open CoverSubjects | Open InteriorCoverSubjects WhereOf
    Cover_Close -> Close CoverSubjects | Close InteriorCoverSubjects WhereOf

    CoverSubjects[NUM=?n, GEN=?g, ART=?a] -> The[NUM=?n, GEN=?g, ART=?a] Cover[NUM=?n, GEN=?g, ART=?a] | Cover[NUM=?n, GEN=?g, ART=?a]
    CoverSubjects[NUM=pl, GEN=?g, ART=?a] -> Every[GEN=?g] The[NUM=pl, GEN=?g, ART=?a] Cover[NUM=pl, GEN=?g, ART=?a]
    InteriorCoverSubjects[NUM=?n, GEN=?g, ART=?a] -> The[NUM=?n, GEN=?g, ART=?a] InteriorCover[NUM=?n, GEN=?g, ART=?a]
    InteriorCoverSubjects[NUM=pl, GEN=?g, ART=?a] -> Every[GEN=?g] The[NUM=pl, GEN=?g, ART=?a] InteriorCover[NUM=pl, GEN=?g, ART=?a]

    Cover[NUM=?n, GEN=?g, ART=?a] -> InteriorCover[NUM=?n, GEN=?g, ART=?a] | ExteriorCover[NUM=?n, GEN=?g, ART=?a]
    InteriorCover[NUM=?n, GEN=?g, ART=?a] -> Blind[NUM=?n, GEN=?g, ART=?a] | Curtain[NUM=?n, GEN=?g, ART=?a]
    InteriorCover[NUM=?n, GEN=?g, ART=?a] -> Door[NUM=?n, GEN=?g, ART=?a] | Shade[NUM=?n, GEN=?g, ART=?a]
    InteriorCover[NUM=?n, GEN=?g, ART=?a] -> Shutter[NUM=?n, GEN=?g, ART=?a] | Window[NUM=?n, GEN=?g, ART=?a]
    ExteriorCover[NUM=?n, GEN=?g, ART=?a] -> Garage[NUM=?n, GEN=?g, ART=?a] | Awning[NUM=?n, GEN=?g, ART=?a]
    ExteriorCover[NUM=?n, GEN=?g, ART=?a] -> Gate[NUM=?n, GEN=?g, ART=?a]

    Climate_Get -> CanYouTell ClimateGetQuestion | ClimateGetQuestion
    ClimateGetQuestion -> WhatIs Temp WhereOf | "che temperatura c'è" Where
    ClimateGetQuestion -> HowHotCold Where | The[NUM=pl, GEN=m, ART=il] TempUnit WhereOf
    HowHotCold -> 'quanto fa' HotCold | 'quanto' HotCold 'fa' | "quanto c'è" HotCold
    HowHotCold -> 'quanto' HotCold "c'è" | 'quanti' TempUnit 'ci sono'
    HotCold -> 'caldo' | 'freddo'

    Climate_Set -> Set Temp Onto Temperature | Set Temp WhereOf Onto Temperature
    Climate_Set -> Set NumTemp TempUnit Where | Set Where Temp 'di' Temperature
    Climate_Set -> Change Temp Into Temperature | Change Temp WhereOf Into Temperature
    Temperature -> NumTemp | NumTemp TempUnit
    NumTemp -> 'quindici' | 'sedici' | 'diciassette' | 'diciotto' | 'diciannove' | 'venti' | 'ventuno'
    NumTemp -> 'ventidue' | 'ventitre' | 'ventiquattro' | 'venticinque'
    Temp ->  The[NUM=sg, GEN=f] 'temperatura'
    TempUnit -> 'gradi' | 'gradi celsius' | 'gradi centigradi'

    """

    gr = grammar.FeatureGrammar.fromstring(g)
    parser = parse.FeatureEarleyChartParser(gr)

    df = pd.DataFrame(
        {
            "Text": [],
            "Intent": [],
            "Domain": [],
            "Name": [],
            "Area": [],
            "DeviceClass": [],
            "Response": [],
            "State": [],
        },
    )

    i = 0
    for s in generate(gr):
        for tree in parser.parse(s):
            i = i + 1
            _LOGGER.info(str(i) + " " + " ".join(s).strip())

            response = "default"
            state = "none"
            names = [
                " ".join(st.leaves())
                for st in list(
                    tree.subtrees(filter=lambda x: "Name" in x.label().values())
                )
            ]
            if names:
                name = names[0]
            else:
                name = "none"

            areas = [
                " ".join(st.leaves())
                for st in list(
                    tree.subtrees(filter=lambda x: "Area" in x.label().values())
                )
            ]
            if areas:
                area = areas[0]
            else:
                area = "none"

            intents = [
                [" ".join(st1.label().values()) for st1 in st]
                for st in list(
                    tree.subtrees(filter=lambda x: "Intent" in x.label().values())
                )
            ]
            intent = intents[0][0]

            domains = [
                [" ".join(st1.label().values()) for st1 in st]
                for st in list(
                    tree.subtrees(
                        filter=lambda x: intent in x.label().values()  # noqa: B023
                    )
                )
            ]
            domain = domains[0][0].split("_")[0].lower()
            if domain not in ("cover", "light", "fan"):
                domain = "none"

            if domain == "cover":
                covers = [
                    [" ".join(st1.label().values()) for st1 in st]
                    for st in list(
                        tree.subtrees(
                            filter=lambda x: "ExteriorCover" in x.label().values()
                            or "InteriorCover" in x.label().values()
                        )
                    )
                ]
                device_class = covers[0][0].split(" ")[0]
            else:
                device_class = "none"

            if intent in ("HassTurnOn", "HassTurnOff"):
                response = turn_onoff_response(area, domain, device_class)
            elif intent == "HassGetState":
                responses = [
                    [" ".join(st1.label().values()) for st1 in st]
                    for st in list(
                        tree.subtrees(
                            filter=lambda x: "Entity_Get" in x.label().values()
                            or "Cover_Get" in x.label().values()
                        )
                    )
                ]
                response = responses[0][0].replace("Cover_", "").lower()
                if domain != "cover":
                    domains = [
                        [" ".join(st1.label().values()) for st1 in st]
                        for st in list(
                            tree.subtrees(
                                filter=lambda x: "OnOffDomain" in x.label().values()
                            )
                        )
                    ]
                    if domains:
                        domain = domains[0][0].split(" ")[0].lower()
                    else:
                        domain = "none"
                states = [
                    [" ".join(st1.label().values()) for st1 in st]
                    for st in list(
                        tree.subtrees(
                            filter=lambda x: "OnOffState" in x.label().values()
                            or "OpenCloseState" in x.label().values()
                        )
                    )
                ]
                if states:
                    state = states[0][0].lower().split("state")[0]
                    if state == "open":
                        state = "on"
                    elif state == "close":
                        state = "off"
                else:
                    state = "none"

            elif intent == "HassLightSet":
                response = domains[0][0].split("_Set")[1].lower()
                if area != "none":
                    response = response + "_area"

            row = pd.DataFrame(
                {
                    "Text": " ".join(s).strip(),
                    "Intent": intent,
                    "Domain": domain,
                    "Name": name,
                    "Area": area,
                    "DeviceClass": device_class,
                    "Response": response,
                    "State": state,
                },
                index=[0],
            )
            df = pd.concat([row, df.loc[:]]).reset_index(drop=True)

    df.to_csv(dataset_file_path)
    return df
