"""Module to generate the commands dataset."""
from functools import reduce
import logging

# from os import path
import re

from nltk import grammar, parse
from nltk.parse.generate import generate
from num2words import num2words
import pandas as pd

from .const import COLORS

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


def extract_slot_value(tree, slot_label, additional_slot_label=None, leaf=False):
    """Extract slot value from tree."""
    if leaf:
        result = [
            [" ".join(st.leaves())]
            for st in list(
                tree.subtrees(filter=lambda x: slot_label in x.label().values())
            )
        ]
    elif additional_slot_label is None:
        result = [
            [" ".join(st1.label().values()) for st1 in st]
            for st in list(
                tree.subtrees(filter=lambda x: slot_label in x.label().values())
            )
        ]
    else:
        result = [
            [" ".join(st1.label().values()) for st1 in st]
            for st in list(
                tree.subtrees(
                    filter=lambda x: slot_label in x.label().values()
                    or additional_slot_label in x.label().values()
                )
            )
        ]

    return [item for sublist in result for item in sublist]


def generate_artificial_dataset(dataset_file_path):
    """Generate smart home commands from a context-free grammar."""
    number_strings = ["'" + str(i) + "'" for i in range(0, 101)]
    numbers1to100 = reduce(lambda x, y: x + " | " + y, number_strings)
    colors = reduce(lambda x, y: x + " | " + y, ["'" + str(i) + "'" for i in COLORS])
    g = (
        """
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
    NumPer -> """
        + numbers1to100
        + """

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

    TurnOn -> 'accendi' | CanYouDo 'accendere' | 'attiva' | CanYouDo 'attivare' | 'attacca' | CanYouDo 'attaccare'
    TurnOff -> 'spegni' | CanYouDo 'spegnere' | 'disattiva' | CanYouDo 'disattivare' | 'stacca' | CanYouDo 'staccare'
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

    One ->  CanYouTell OneQuestion | OneQuestion
    OneQuestion -> 'qual è lo stato' Of[NUM=?n, GEN=?g, ART=?a] Name | 'qual è il valore' Of[NUM=?n, GEN=?g, ART=?a] Name

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

    EntitySubject[NUM=?n, GEN=?g, ART=?a] -> The[NUM=sg, GEN=?g, ART=?a] Name[NUM=sg, GEN=?g, ART=?a]

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

    CoverSubject[NUM=?n, GEN=?g, ART=?a] -> The[NUM=?n, GEN=?g, ART=?a] Cover[NUM=?n, GEN=?g, ART=?a]
    InteriorCoverSubject[NUM=?n, GEN=?g, ART=?a] -> The[NUM=?n, GEN=?g, ART=?a] InteriorCover[NUM=?n, GEN=?g, ART=?a]

    Light_TurnOn -> TurnOn LightSubject | TurnOn LightSubject WhereOf
    Light_TurnOff -> TurnOff LightSubject | TurnOff LightSubject WhereOf

    Light_SetBrightness -> Set Brightness WhereOf Onto Percentage | Set Onto Percentage Brightness WhereOf
    Light_SetBrightness -> Set Brightness Onto Percentage | Set Onto Percentage Brightness
    Light_SetBrightness -> Change Brightness Into Percentage | Change Into Percentage Brightness
    Light_SetBrightness -> Change Brightness WhereOf Into Percentage | Change Into Percentage Brightness WhereOf
    Brightness ->  The[NUM=sg, GEN=f] 'luminosità' Of[NUM=?n, GEN=f] Light[NUM=?n, GEN=f] | The[NUM=sg, GEN=f] 'luminosità' Of[NUM=?n, GEN=?g, ART=?a] Name

    Light_SetColor ->  Set Color Onto ColorValue | Set Onto ColorValue Color
    Light_SetColor ->  Change Color Into ColorValue | Change Into ColorValue Color
    Light_SetColor ->  Set Color WhereOf Onto ColorValue | Set Onto ColorValue Color WhereOf
    Light_SetColor ->  Change Color WhereOf Into ColorValue | Change Into ColorValue Color WhereOf
    Color ->  The[NUM=sg, GEN=m, ART=il] 'colore' Of[NUM=?n, GEN=f] Light[NUM=?n, GEN=f] | The[NUM=sg, GEN=m, ART=il] 'colore' Of[NUM=?n, GEN=?g, ART=?a] Name
    ColorValue -> """
        + colors
        + """

    LightSubject[NUM=?n, GEN=f] -> The[NUM=?n, GEN=f] Light[NUM=?n]
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

    Climate_Get -> CanYouTell ClimateGetQuestion | ClimateGetQuestion  |  CanYouTell The[NUM=pl, GEN=m, ART=il] TempUnit WhereOf
    ClimateGetQuestion -> WhatIs Temp WhereOf | "che temperatura c'è" Where | HowHotCold Where
    HowHotCold -> 'quanto fa' HotCold | 'quanto' HotCold 'fa' | "quanto c'è" HotCold
    HowHotCold -> 'quanto' HotCold "c'è" | 'quanti' TempUnit 'ci sono'
    HotCold -> 'caldo' | 'freddo'

    Climate_Set -> Set Temp Onto Temperature | Set Onto Temperature Temp
    Climate_Set -> Set Temp WhereOf Onto Temperature | Set Onto Temperature Temp
    Climate_Set -> Set NumTemp TempUnit Where | Set Where Temp 'di' Temperature
    Climate_Set -> Change Temp Into Temperature | Change Into Temperature Temp
    Climate_Set -> Change Temp WhereOf Into Temperature | Change Into Temperature Temp WhereOf

    Climate[NUM=?n, GEN=?g, ART=?a]  -> Name[NUM=?n, GEN=?g, ART=?a]
    Climate[NUM=sg, GEN=m, ART=il] -> 'riscaldamento' | 'termostato'
    Climate[NUM=sg, GEN=f] -> 'valvola termostatica'
    Temperature -> NumTemp | NumTemp TempUnit | NumTemp 'e mezzo' | NumTemp TempUnit 'e mezzo'
    NumTemp -> '15' | '16' | '17' | '18' | '19' | '20' | '21'
    NumTemp -> '22' | '23' | '24' | '25'
    Temp ->  The[NUM=sg, GEN=f] 'temperatura' | The[NUM=sg, GEN=f] 'temperatura' Of[NUM=?n, GEN=?g, ART=?a] Climate[NUM=?n, GEN=?g, ART=?a]
    TempUnit -> 'gradi' | 'gradi celsius' | 'gradi centigradi'

    """
    )

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
            "Color": [],
            "Brightness": [],
            "Temperature": [],
        },
    )

    i = 0
    for s in generate(gr):
        for tree in parser.parse(s):
            i = i + 1
            text = re.sub(
                r"(\d+)",
                lambda x: num2words(int(x.group(0)), lang="it"),
                " ".join(s).strip(),
            )

            response = "default"
            state = "none"
            brightness = "none"
            color = "none"
            name = "none"
            area = "none"
            device_class = "none"
            state = "none"
            temperature = "none"

            names = extract_slot_value(tree, "Name", leaf=True)
            if names:
                name = names[0]

            areas = extract_slot_value(tree, "Area", leaf=True)
            if areas:
                area = areas[0]

            intents = extract_slot_value(tree, "Intent")
            intent = intents[0]

            domains = extract_slot_value(tree, intent)
            domain = domains[0].split("_")[0].lower()
            if domain not in ("cover", "light", "fan", "climate"):
                domain = "none"

            if domain == "cover":
                covers = extract_slot_value(tree, "ExteriorCover", "InteriorCover")
                device_class = covers[0].split(" ")[0]

            if intent in ("HassTurnOn", "HassTurnOff"):
                response = turn_onoff_response(area, domain, device_class)
            elif intent == "HassGetState":
                responses = extract_slot_value(tree, "Entity_Get", "Cover_Get")
                response = responses[0].replace("Cover_", "").lower()
                if domain != "cover":
                    domains = extract_slot_value(tree, "OnOffDomain")
                    if domains:
                        domain = domains[0].split(" ")[0].lower()
                states = extract_slot_value(tree, "OnOffState", "OpenCloseState")
                if states:
                    state = states[0].lower().split("state")[0]
                    if state == "open":
                        state = "on"
                    elif state == "close":
                        state = "off"

            elif intent == "HassLightSet":
                response = domains[0].split("_Set")[1].lower()
                if response == "brightness":
                    numbers = extract_slot_value(tree, "NumPer", leaf=True)
                    brightness = numbers[0]
                else:
                    colors = extract_slot_value(tree, "ColorValue", leaf=True)
                    color = colors[0]
                if area != "none":
                    response = response + "_area"
            elif intent == "HassClimateSetTemperature":
                temperature = extract_slot_value(tree, "NumTemp", leaf=True)[0]
                if "e mezzo" in text:
                    temperature = temperature + ".5"

            if text.startswith(
                ("puoi", "potresti", "qual", "quant", "com", "c'è", "ci sono")
            ):
                text = text + "?"

            _LOGGER.info(str(i) + " " + text)
            # print(str(i) + " " + text)

            row = pd.DataFrame(
                {
                    "Text": text.lower(),
                    "Intent": intent,
                    "Domain": domain,
                    "Name": name,
                    "Area": area,
                    "DeviceClass": device_class,
                    "Response": response,
                    "State": state,
                    "Color": color,
                    "Brightness": brightness,
                    "Temperature": temperature,
                },
                index=[0],
            )
            df = pd.concat([row, df.loc[:]]).reset_index(drop=True)

    df.to_csv(dataset_file_path)
    return df


# generate_artificial_dataset(path.join(DATASETS_DIR, "HassGetState.csv"))
