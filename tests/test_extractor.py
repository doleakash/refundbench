from app.domain.models import GrievanceType
from app.evaluation.extractor import GrievanceExtractor


def test_extract_multiple_grievances():
    extractor = GrievanceExtractor()

    message = (
        "My order was 85 minutes late, "
        "2 of my 4 biryanis were missing, "
        "and the raita container leaked."
    )

    grievances = extractor.extract(message)

    assert len(grievances) == 3

    assert grievances[0].type == GrievanceType.LATE_DELIVERY
    assert grievances[1].type == GrievanceType.MISSING_ITEMS
    assert grievances[2].type == GrievanceType.LEAKED_ITEM