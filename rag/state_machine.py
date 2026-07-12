from enum import Enum
class LoanStage(Enum):
    START="start"
    CATEGORY="category"
    AMOUNT="amount"
    TERM="term"
    DOCUMENT="document"
    ELIGIBILITY="eligibility"
    PENDING="pending"
    COMPLETE="complete"

class LoanStateMachine:
    def next_stage(
        self,
        current_stage,
        entities
    ):
        if current_stage == LoanStage.START:

            if "loan_category" not in entities:

                return LoanStage.CATEGORY

        if current_stage == LoanStage.CATEGORY:

            if "amount" not in entities:

                return LoanStage.AMOUNT

        if current_stage == LoanStage.AMOUNT:

            if "term_months" not in entities:

                return LoanStage.TERM

        if current_stage == LoanStage.TERM:

            return LoanStage.DOCUMENT

        if current_stage == LoanStage.DOCUMENT:

            if "documents" not in entities:

                return LoanStage.DOCUMENT

        if current_stage == LoanStage.ELIGIBILITY:

            return LoanStage.PENDING
        return LoanStage.COMPLETE