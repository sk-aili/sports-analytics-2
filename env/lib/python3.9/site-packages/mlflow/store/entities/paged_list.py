from typing import List, Optional, TypeVar

T = TypeVar("T")


class PagedList(List[T]):
    """
    Wrapper class around the base Python `List` type. Contains an additional `token`  string
    attribute that can be passed to the pagination API that returned this list to fetch additional
    elements, if any are available
    """

    # TODO (chiragjn): Make total compulsory by refactoring all usages!
    def __init__(self, items: List[T], token, total: Optional[int] = None):
        super().__init__(items)
        self.total = total
        self.token = token
