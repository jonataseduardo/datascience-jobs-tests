from pathlib import Path
import xml.etree.ElementTree as ET


def get_abstract_from_xml(xml_string):
    root = ET.parse(xml_string)
    abstract = root.find("Award/AbstractNarration").text
    title = root.find("Award/AwardTitle").text
    return [title, abstract]


files = sorted(Path("../data").rglob("*.xml"))

text = [[f] + get_abstract_from_xml(str(f)) for f in files]

text[-1]
