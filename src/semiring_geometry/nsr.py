"""
This contains classes and methods related to the 
Storage-Limited Store-and-Forward Nevada Semiring.
"""

from __future__ import annotations

from typing import Self

import portion as P


INF = float("inf")

def intersect(intervals: list[P.Interval]) -> P.Interval:
    """
    Returns the intersection of the intervals in `intervals`.

    Parameters
    ----------
    intervals : list[P.Interval]
        A list of portion interval objects.

    Returns
    -------
    P.interval
        A single portion interval object which is the intersection of all 
        the elements of `intervals`.
    """
    interval = P.closed(-INF, INF)
    
    for i in intervals:
        interval &= i

    return interval

def contains(interval: P.Interval, k: float) -> bool:
    """
    Checks if the value `k` is contained within the `interval`.

    Parameters
    ----------
    interval : P.Interval
        A portion interval object.
    k : float
        A value to check.

    Returns
    -------
    bool
        True if `k` is within the `interval`, otherwise False.
    """
    if k == -INF:
        return interval.lower == k
    if k == INF:
        return interval.upper == k
    return k in interval

def get_ascii_diagram(
    element: Contact | Storage | Nevada, 
    start: float, 
    end: float,
    step: float = 1
) -> str:
    """
    Generates an ASCII diagram representation of the given element over a 
    given window. 

    Works on any of the classes here implementing the `get_entry` method.

    Parameters
    ----------
    element : Contact | Storage | Nevada
        The object to generate the diagram of.
    start : float
        The starting value of the range.
    end : float
        The ending value of the range.
    step : float, optional
        The step size for the range (default is 1).

    Returns
    -------
    str
        A string representing the ASCII diagram of the element over the window.
    """
    indices = [round(start + delta * step, 3) for delta in range(int((end - start) / step) + 1)]

    diagram = ""
    for i in indices:
        for j in indices:
            value = f"{element.get_entry(i, j)} "

            if element.get_entry(i, j) > 0:
                value = "*"
            else:
                value = "-"

            diagram += value
        diagram += "\n"

    return diagram

if __name__ == "__main__":
    intervals = [P.closed(0, 5), P.closed(3, 6)]
    assert intersect(intervals) == P.closed(3, 5)

    intervals += [P.closed(4, 5)]
    assert intersect(intervals) == P.closed(4, 5)
    
    intervals += [P.closed(7, INF)]
    assert intersect(intervals) == P.empty()

    assert contains(P.closed(0, 5), 5)
    assert not contains(P.closed(0, 5), 6)

    assert contains(P.closed(0, INF), 0)
    assert contains(P.closed(0, INF), INF)
    assert not contains(P.closed(0, INF), -INF)

    assert contains(P.closed(-INF, INF), 5)
    assert contains(P.closed(-INF, INF), INF)
    assert contains(P.closed(-INF, INF), -INF)

Point = tuple[float, float]

def isinstance_point(obj: object) -> bool:
    """
    Checks if the object is a point represented as a tuple of two numbers.

    Parameters
    ----------
    obj : object
        The object to check.

    Returns
    -------
    bool
        True if the object is a tuple of two numbers, otherwise False.
    """
    if isinstance(obj, tuple):
        types = list(map(type, obj))
        return types == [int, int] or types == [float, float] or \
            types == [int, float] or types == [float, int]
    return False

if __name__ == "__main__":
    assert isinstance_point((1, 2))
    assert isinstance_point((1, 2.0))
    assert isinstance_point((1.0, 2.0))
    assert not isinstance_point((0, "2.0"))
    assert not isinstance_point("test")
    assert not isinstance_point(0)


class Contact():

    def __init__(self, 
        start: float, 
        end: float, 
        delay: float
    ) -> None:

        # [start, end] must form interval
        if start > end:
            message = f"`start` must be less than `end` ({start}, {end})"
            raise ValueError(message)

        # start in [-inf, inf)
        if start == INF:
            message = f"must have `-inf <= start < inf` ({start}, {end})"
            raise ValueError(message)

        # end in (-inf, inf]
        if end == -INF:
            message = f"must have `-inf < end <= inf` ({start}, {end})"
            raise ValueError(message)
        
        # delay in [0, inf)
        if delay < 0 or delay == INF:
            message = f"must have `0 <= delay < INF ({delay})"
            raise ValueError(message)

        self.start = start
        self.end = end
        self.delay = delay

    def get_interval(self) -> P.Interval:
        """
        Returns the interval defined by the `start` and `end` attributes of 
            the Contact.

        Returns
        -------
        P.Interval
            A closed portion interval between `self.start` and `self.end`.
        """
        return P.closed(self.start, self.end)

    def contains_point(self, i: float, j: float) -> bool:
        """
        Checks if the index [i, j] lies within the support of the associated
        matrix.
        
        Parameters
        ----------
        i : float
            A row index of the matrix.
        j : float
            A column index of the matrix.
        
        Returns
        -------
        bool
            True if [i, j] lies within the support of the associated matrix.
            Otherwise, returns False.
        """
        if self.start <= i and i <= self.end and j == i + self.delay:
            return True
        return False

    def __getitem__(self, index: Point) -> bool | float:
        """
        Retrieves the [i, j]-th entry of the matrix.
        This defauls to the [0, infty]-semiring, so the result will be a float.

        Parameters
        ----------
        index : Point
            A tuple representing the (i, j)-th entry of the matrix.

        Returns
        -------
        bool | float
            The value of the entry at the [i, j]-th coordinates. 
            Will be zero if it lies outside the support.
        """
        return self.get_entry(index[0], index[1])

    def get_entry(self, i: float, j: float, flag: bool = False) -> bool | float:
        """
        Retrieves the [i, j]-th entry of the matrix.
        
        If `flag` is True, it assumes you're working within the boolean 
        semiring and returns boolean values.

        If `flag` is False, it assumes you're working within the [0, infty]
        semiring and returns real values [0, infty].

        Parameters
        ----------
        i : float
            A row index of the matrix.
        j : float
            A column index of the matrix.
        flag : bool, optional
            If True, assumes you're working within the boolean semiring.
            If False assumes you're working within the [0, infty] semiring.

        Returns
        -------
        bool | float
            The value of the matrix at the [i, j]-th entry in the specified 
            semiring.
        """
        contained = self.contains_point(i, j)

        if flag:
            return contained
        return self.end - i if contained else 0

        # if self.contains_point(i, j):
        #     return True if self.boolean else self.end - i
        # return False if self.boolean else 0

    def get_boundary(self) -> list[Point]:
        """
        Calculates the boundary points of the Contact.
    
        Returns
        -------
        list[Point]
            A list of unique boundary points, where each point is a tuple 
            (i, j) representing the (row, column) indices of the 
            associated matrix.
        """
        points = [
            (self.start, self.start + self.delay),
            (self.end, self.end + self.delay)
        ]
        return list(set(points))

    def __eq__(self, other: object) -> bool:
        """
        Checks if the current `Contact` instance is equal to another object.

        Parameters
        ----------
        other : object
            The object to compare with the current instance. 

        Returns
        -------
        bool
            True if `other` is a `Contact` instance with the same
            `start`, `end` and `delay. Otherwise, returns False.
        """
        if isinstance(other, Contact):
            return self.delay == other.delay and \
                self.get_interval() == other.get_interval()
        return False

    def __contains__(self, other: Contact | Point | object) -> bool:
        """
        Checks if the specified object is contained within the Contact.

        Parameters
        ----------
        other : Contact | Point | object
            The object to check for containment.
            This can be a tuple representing an index, 
            another `Contact` instance, or any other object.

        Returns
        -------
        bool
            True if the object is contained within the instance, 
            otherwise False.
        """
        if isinstance(other, tuple) and isinstance_point(other):
            return self.contains_point(*other)

        if isinstance(other, Contact):
            if self.delay != other.delay or \
                other.get_interval() not in self.get_interval():
                return False
            return True

        return False

    def __add__(self, other: Contact) -> Sum:
        """
        Adds two `Contact` instances, returning a `Sum` object.

        Parameters
        ----------
        other : Contact
            Another `Contact` instance to add to the current instance.

        Returns
        -------
        Sum
            A `Sum` object containing one or both `Contact` instances:
            If either is contained within the other, it returns the larger of 
            the two.
        """
        if self in other:
            return Sum([other])
        if other in self:
            return Sum([self])
        return Sum([self, other])

    def __mul__(self, other: Contact) -> Contact:
        """
        Performs multiplication of two `Contact` instances.

        Parameters
        ----------
        other : Contact
            Another `Contact` instance to multiply with the current instance.

        Returns
        -------
        Contact
            A new `Contact` instance representing the product.
            
            If the resulting interval is invalid (i.e., `end - start < 0`),
            the resulting `Contact` will have `start`, `end`, and `delay` zero.

        Raises
        ------
        NotImplemented
            If `other` is not an instance of `Contact`.
        """
        if not isinstance(other, Contact):
            return NotImplemented
            
        start = max(self.start, other.start - self.delay)
        end = min(self.end, other.end - self.delay)
        delay = self.delay + other.delay

        if end - start < 0:
            start, end, delay = 0, 0, 0

        return Contact(start, end, delay)

    def __format__(self, spec: str) -> str:
        """
        Formats the `Contact` instance as a string, with an option to format it
        in LaTeX code.

        Parameters
        ----------
        spec : str
            The format specification. 
            - If `spec` is "t", the output is formatted as valid LaTeX code.
            - Otherwise, it defaults to the string representation of the object.

        Returns
        -------
        str
            The formatted string representation of the `Contact` instance.
            If `spec` is "t", it returns a LaTeX-compatible string with the 
            interval and delay.
            Otherwise, it returns the default string representation.
        """
        if spec == "t":
            start = str(self.start)
            end = str(self.end)
            if self.start == -INF:
                start = "-\\infty"
            if self.end == INF:
                end = "\\infty"

            return f"([{start}, {end}] : {self.delay})"
        return str(self)

    def __str__(self) -> str:
        return f"([{self.start}, {self.end}] : {self.delay})"

    @staticmethod
    def identity() -> Contact:
        return Contact(-INF, INF, 0)


if __name__ == "__main__":

    # test __contains__
    assert Contact(1, 2, 5) in Contact(0, 3, 5)
    assert (5, 5) in Contact(0, 6, 0)
    assert (7, 7) not in Contact(0, 6, 0)

    # test get_entry()
    assert Contact(4, 6, 0).get_entry(5, 5)
    assert not Contact(4, 6, 0).get_entry(7, 7)

    # test get_boundary()
    points = Contact(4, 6, 0).get_boundary()
    assert (4, 4) in points and (6, 6) in points

    # test get_boundary() on infinite contact
    points = Contact(-INF, INF, 5).get_boundary()
    assert (-INF, -INF) in points and (INF, INF) in points

    assert Contact(5, 5, 0).get_entry(5, 5, True)
    # print()

    # points = [(0, 1), (0, 1)]
    # points = list(set(points))
    # print(f"{points = }")

    # diagram = get_ascii_diagram(Contact(4, 6, 0), 2, 8, 0.2)
    # print(diagram)

class Storage():
    def __init__(self, capacity: float = INF):

        # capacity must be in [0, inf]
        if capacity < 0:
            message = f"must have `0 <= capacity <= INF ({capacity})"
            raise ValueError(message)

        self.capacity = capacity

    def __getitem__(self, index: Point) -> bool | float:
        return self.get_entry(index[0], index[1])

    def get_entry(self, i: float, j: float, flag: bool = False) -> bool | float:

        contained = self.contains_point(i, j)

        if flag:
            return contained

        return INF if contained else 0

        # if i <= j and j <= i + self.capacity:
        #     return True if self.boolean else INF
        # return False if self.boolean else 0

    def get_boundary(self) -> list[Point]:
        points = [
            (-INF, -INF), # top left of matrix 
            (INF, INF), # bottom right of matrix
        ]

        if self.capacity == INF:
            points.append((-INF, INF)) # top right of matrix
        # else:
            # each point has a mulitplicity of two
            # with the understanding that one of the parallel lines is offset
            # points += points 

        return points

    def contains_point(self, i: float, j: float) -> bool:
        if i <= j and j <= i + self.capacity:
            return True

        return False

    def contains_contact(self, contact: Contact) -> bool:
        return contact.delay <= self.capacity

    def __eq__(self, other) -> bool:
        if isinstance(other, Storage):
            return self.capacity == other.capacity

        return False

    def __contains__(self, other: Storage | Contact | Point | object) -> bool:

        if isinstance(other, tuple) and isinstance_point(other):
            return self.contains_point(*other)

        if isinstance(other, Contact):
            return self.contains_contact(other)

        if isinstance(other, Storage):
            return self.capacity >= other.capacity

        return False

    def __add__(self, other: Storage | Contact) -> Sum:
        if other in self:
            return Sum([self])
        if self in other:
            return Sum([other])
        return Sum([self, other])

    def __radd__(self, other: Contact) -> Sum:
        if other in self:
            return Sum([self])
        return Sum([other, self])

    def __mul__(self, other: Storage | Contact) -> Nevada | Storage:

        if isinstance(other, Contact):
            return Nevada(Contact.identity(), other, self)
        
        if isinstance(other, Storage):
            return Storage(self.capacity + other.capacity)

        return NotImplemented

    def __rmul__(self, other: Contact) -> Nevada:
        if isinstance(other, Contact):
            return Nevada(other, Contact.identity(), self)
        return NotImplemented

    def __format__(self, spec: str) -> str:
        if spec == "t":
            if self.capacity == INF:
                capacity = "\\infty"
            else:
                capacity = str(self.capacity)
            return "S_{" + capacity + "}"
        return str(self)

    def __str__(self) -> str:
        return f"S_({self.capacity})"

    @staticmethod
    def identity() -> Storage:
        return Storage(capacity=0) # TODO : double check

if __name__ == "__main__":

    # test get_entry() [max-min sr]
    assert Storage(5).get_entry(5, 5) == INF
    assert Storage(5).get_entry(5, 15) == 0

    # test get_entry() [boolean sr]
    assert Storage(5).get_entry(5, 5, True) == True
    assert Storage(5).get_entry(5, 15, False) == False

    # test __contains__
    assert Storage(5) in Storage(5)
    assert Storage(5) in Storage()
    assert Storage(5) not in Storage(3)

    assert Contact(0, 5, 0) in Storage()
    assert (0, 0) in Storage() and (5, 5) in Storage()
    assert Contact(0, 5, 5) not in Storage(2)
    assert (0, 3) not in Storage(2)

class Nevada():

    def __init__(self, *args: Contact | Storage) -> None:

        if len(args) == 1 and isinstance(args[0], Contact):
            (l, r, s) = Nevada.standard_form(
                Contact.identity(), 
                args[0], 
                Storage(0)
            )
            # self.left = Contact.identity()
            # self.right = args[0]
            # self.storage = Storage(0)
            (self.left, self.right, self.storage) = (l, r, s)
        elif len(args) == 1 and isinstance(args[0], Storage):
            (l, r, s) = Nevada.standard_form(
                Contact.identity(), 
                Contact.identity(), 
                args[0]
            )
            # self.left = Contact.identity(boolean)
            # self.right = Contact.identity(boolean)
            # self.storage = args[0]
            (self.left, self.right, self.storage) = (l, r, s)
        # elif len(args) == 1 and isinstance(args[0], ContactSequenceSummary):
        # elif len(args) == 1 and isinstance(args[0], ContactSequence):
        elif len(args) == 3 and isinstance(args[0], Contact) and \
            isinstance(args[1], Contact) and isinstance(args[2], Storage):

            (l, r, s) = Nevada.standard_form(args[0], args[1], args[2])
            (self.left, self.right, self.storage) = (l, r, s)

        else:
            raise ValueError("Invalid arguments to initialize `Nevada`")

    @staticmethod
    def standard_form(
        left: Contact,
        right: Contact,
        storage: Storage
    ) -> tuple[Contact, Contact, Storage]:

        left_contact = Contact(0, 0, 0)
        if left.start <  min(left.end, right.end - left.delay):
            left_contact = Contact(
                # left.start,
                max(left.start, right.start - left.delay - storage.capacity),
                min(left.end, right.end - left.delay),
                0
            )

        right_contact = Contact(0, 0, 0)
        if max(left.start, right.start - left.delay) < right.end - left.delay:
            right_contact = Contact(
                max(left.start, right.start - left.delay),
                # right.end - left.delay,
                min(left.end + storage.capacity, right.end - left.delay),
                left.delay + right.delay
            )

        standard_form = (
            left_contact, 
            right_contact, 
            Storage(storage.capacity)
        )

        return standard_form

    def __getitem__(self, index: Point) -> bool | float:
        return self.get_entry(index[0], index[1])

    def get_entry(self, i: float, j: float, flag: bool = False) -> bool | float:

        # if (i, j) in self:
        if self.contains_point(i, j):
            value = min(self.left.end - i, self.right.end - j + self.right.delay)
            return True if flag else value

        return False if flag else 0

    def get_boundary(self) -> list[Point]:
        """
        This function returns the boundary points of the matrix.

            A . B       BC
            .      .
            .         .
            F           C
              .         .
                 .      .
                    E . D
        """

        # if self.is_storage():
        #     message = "`get_boundary` doesn't work with Storage-type Nevada"
        #     raise TypeError(message)
        
        left = self.left
        right = self.right
        capacity = self.storage.capacity

        # A : intersection of : {horizontal line s1} and {vertcal line s2 + w2}
        #       A = (s1, s2 + w2)
        # A : if intersection z := {x = y + Omega} & {horizontal line s1} =
        #       is to right of {vertical line s2 + w2}, use z
        #       A = (s1, s1 + w1 + w2)
        # A : if intersection z := {x = y + omega + alpha} & {vertical line s2 + w2}
        #       is below horizontal line s1,
        #       A = (s2 - w1 - alpha, s2 + w2)
        A = (max(left.start, right.start - left.delay - capacity), 
            max(right.start + right.delay, 
                left.start + left.delay + right.delay))

        # A : intersection of horizontal line s1 and vertical line s2 + w2
        # A = (max(left.start, right.start - left.delay - capacity), 
        # #         right.start + right.delay)

        # B : if alpha is limited, this is the intersection of
        #       horizontal : line s1
        #       diagonal line : x = y + omega + alpha

        D = (min(left.end, right.end - left.delay), 
            min(right.end + right.delay, 
                left.end + left.delay + right.delay + capacity))
        E = (min(left.end, right.end - left.delay), 
            min(left.end + left.delay + right.delay, 
                right.end + right.delay))
        # EF can never exist; would go below the diagonal, which is zero
        F = (max(right.start - left.delay, left.start), 
            max(right.start + right.delay, 
                left.start + left.delay + right.delay))

        boundary = set([A, D, E, F])
        if capacity == INF:
            BC = (left.start, right.end + right.delay)

            boundary.add(BC)
        else:
            B = (max(left.start, right.start - left.delay - capacity), 
                min(left.start + left.delay + right.delay + capacity, 
                    right.end + right.delay))
            C = (min(left.end, right.end - left.delay - capacity), 
                min(right.end + right.delay, 
                    left.end + left.delay + right.delay + capacity))

            boundary.update([B, C])



        # points = {}
        # boundary = []
        # for p in ["A", "B", "C", "D", "E", "F"]:
        #     # points[p] = locals()[p]
        #     boundary.append(locals()[p])
        #     print(f"{p} : {locals()[p]}")
        

        # boundary = [A, D, E, F]

        # maybe different based on alpha = inf or alpha < inf
        # if capacity == INF:
        #     # unlimited storage
        #     # print("unlimited storage")
        #     boundary.append(BC)
        # else:
        #     # limited storage
        #     # print("limited storage")
        #     boundary.append(B)
        #     boundary.append(C)

        return list(boundary)

    def is_storage(self) -> bool:

        return self.left == Contact.identity() and \
            self.right == Contact.identity()

    def contains_point(self, i: float, j: float) -> bool:

        # Proposition 6.5

        # handle infinities
        # if (i, j) in [(-INF, -INF), (-INF, INF), (INF, INF)]:
        #     return (i, j) in self.get_boundary()

        if {i, j} & {-INF, INF}:
            message = f"The indices of the matrix can't be infinite : {i}, {j}"
            raise ValueError(message)

        delay_shift_l = lambda x : x - self.left.delay - self.storage.capacity
        delay_shift_u = lambda x : x - self.left.delay 
        interval_a = self.left.get_interval() & \
            self.right.get_interval().replace(
                lower = delay_shift_l, 
                upper = delay_shift_u
            )
        condition_a: bool = i in interval_a
        # condition_a: bool = contains(interval_a, i)

        delay_shift_l = lambda x : x + self.left.delay 
        delay_shift_u = lambda x : x + self.left.delay + self.storage.capacity
        interval_b = self.right.get_interval() & \
            self.left.get_interval().replace(
                lower = delay_shift_l, 
                upper = delay_shift_u
            )
        condition_b: bool = (j - self.right.delay) in interval_b
        # condition_b: bool = contains(interval_b, j - self.right.delay)

        delay = self.left.delay + self.right.delay
        interval_c = P.closed(i + delay, i + delay + self.storage.capacity)
        condition_c: bool = j in interval_c
        # condition_c: bool = contains(interval_c, j)
        # condition_c = i + delay <= j and j <= i + delay + self.storage.capacity

        return condition_a and condition_b and condition_c

    def contains_contact(self, other: Contact) -> bool:

        alpha = self.storage.capacity
        omega = self.left.delay + self.right.delay
        if other.delay < omega or other.delay > omega + alpha:
            return False

        points = other.get_boundary()

        for p in points:
            if p in [(-INF, -INF), (INF, INF)]:
                if p not in self.get_boundary():
                    return False
                continue
            if not self.contains_point(*p):
                return False

        return True

    def contains_storage(self, other: Storage) -> bool:

        if self.is_storage():
            return other in self.storage

        return False

    def contains_nevada(self, other: Nevada) -> bool:
        # I think this one is more complicated since the get_boundary() 
        #   method might not be enough. If you have to parallel lines meeting 
        #   at infinity you could have problems.

        # case: `other` is of type `Storage`
        if other.is_storage():
            return self.contains_storage(other.storage)

        # case: `self` is of type `Storage` and `other` is not
        if self.is_storage():
            omega = other.left.delay + other.right.delay
            return omega + other.storage.capacity <= self.storage.capacity

        # case: both `self` and `other` are not of type `Storage`
        points = other.get_boundary()

        for p in points:
            if not self.contains_point(*p):
                return False
        return True

    def __contains__(self, other: Nevada | Storage | Contact | Point) -> bool:

        if isinstance(other, tuple) and isinstance_point(other):
            return self.contains_point(*other)

        if isinstance(other, Contact):
            # print(f"Checking if Nevada contains a Contact")
            return self.contains_contact(other)

        if isinstance(other, Storage):
            return self.contains_storage(other)

        if isinstance(other, Nevada):
            return self.contains_nevada(other)

        return False

    def __eq__(self, other) -> bool:
        # assumes both self and other are in standard form
        return self.left == other.left and self.right == other.right and \
            self.storage == other.storage

    def __add__(self, other: Nevada | Storage | Contact) -> Sum:

        if other in self:
            return Sum([self])
        if self in other:
            return Sum([other])
        return Sum([self, other])

    def __radd__(self, other: Storage | Contact) -> Sum:
        if other in self:
            return Sum([self])
        if self in other:
            return Sum([other])
        return Sum([other, self])

    def __mul__(self, other: Nevada | Storage | Contact) -> Nevada:
                
        if isinstance(other, Nevada):
            # both are nevadas

            ic = self.right * other.left # calculate inner contact

            c = Contact(
                ic.start - self.storage.capacity, 
                ic.end, 
                0
            )
            left = self.left * c

            c = Contact(
                ic.start, 
                ic.end + other.storage.capacity, 
                ic.delay
            )
            right = c * other.right

            storage = Storage(
                self.storage.capacity + other.storage.capacity
            )

            return Nevada(left, right, storage)
        elif isinstance(other, Storage):
            # other is storage; cast to nevada and use above logic
            return self * Nevada(other)
        elif isinstance(other, Contact):
            return Nevada(self.left, self.right * other, self.storage)
        else:
            return NotImplemented

    def __rmul__(self, other: Contact | Storage) -> Nevada:

        if isinstance(other, Storage):
            return Nevada(other) * self
        elif isinstance(other, Contact):
            return Nevada(other * self.left, self.right, self.storage)
        else:
            return NotImplemented

    def __format__(self, spec: str) -> str:
        if spec == "t":
            return f"{self.left:t} \\cdot {self.storage:t} \\cdot {self.right:t}"
        return str(self)


    def __str__(self) -> str:
        return f"{self.left} {self.storage} {self.right}"

    def get_ascii_diagram(self, start: float, end: float, step: float = 1):
        return get_ascii_diagram(self, start, end, step) 

if __name__ == "__main__":
    c = Contact.identity()
    s = Storage()
    ns = Nevada(Storage())
    
    assert Nevada(c, c, s) == ns

    s_five = Storage(5)
    ns_limited = Nevada(s_five)
    assert Nevada(c, c, s_five) == ns_limited

    assert Nevada(c, c, s).contains_contact(Contact(1, 2, 3))
    assert Contact(1, 2, 3) in ns
    assert Nevada(c, c, s).is_storage()
    assert Nevada(c, c, Storage(3)).is_storage()

    assert not Nevada(c, Contact(1,5, 0), s).is_storage()

    # test standard form conversion works
    assert Nevada(Contact(2, 6, 1), Contact(4, 8, 1), s) == Nevada(Contact(2, 6, 0), Contact(3, 7, 2), s)
    assert Nevada(Contact(2, 6, 1), Contact(4, 8, 1), Storage(2)) == Nevada(Contact(2, 6, 0), Contact(3, 7, 2), Storage(2))

    # boundary c * S_(inf) 
    assert set([(2, 2), (8, 8), (2, INF), (8, INF)]) == set(Nevada(Contact(2, 8, 0), c, Storage()).get_boundary())

    # boundary S_(inf) * c
    assert set([(2, 2), (8, 8), (-INF, 2), (-INF, 8)]) == set(Nevada(c, Contact(2, 8, 0), Storage()).get_boundary())

    # boundary c * S_(inf) * c'
    assert set([(6, 8), (2, 9), (2, 5), (6, 9), (3, 5)]) == set(Nevada(Contact(2, 6, 1), Contact(4, 8, 1), Storage()).get_boundary())

    # boundary S_alpha * c; alpha < inf
    assert set([(8, 9), (0, 3), (2, 3), (6, 9)]) == set(Nevada(c, Contact(2, 8, 1), Storage(2)).get_boundary())

    # boundary c * S_alpha; alpha < inf
    assert set([(2, 3), (8, 9), (2, 5), (8, 11)]) == set(Nevada(Contact(2, 8, 1), c, Storage(2)).get_boundary())

    # boundary c * S_alpha * c'; alpha < inf
    # n = Nevada(Contact(2, 6, 0), Contact(3, 7, 2), Storage(2))
    # get_heat_map(n, 1, 10, 0.05)
    assert set([(6, 8), (2, 6), (5, 9), (2, 5), (6, 9), (3, 5)]) == set(Nevada(Contact(2, 6, 0), Contact(3, 7, 2), Storage(2)).get_boundary())

    assert set([(-INF, -INF), (INF, INF)]) == set(Nevada(Storage(2)).get_boundary())
    assert set([(-INF, -INF), (INF, INF), (-INF, INF)]) == set(Nevada(Storage()).get_boundary())

    assert set([(0, 2), (1, 3)]) == set(Nevada(Contact(0, 1, 2)).get_boundary())

    # test __contains__
    assert (0, 0) in ns
    assert (5, 0) not in ns
    assert (0, 15) not in ns_limited

    # print(f"{c}")
    # print(f"{ns}")
    # print(get_ascii_diagram(c, 1, 10, step=0.5))
    assert c in ns
    assert Contact(0, 1, 10) not in ns_limited
    assert Contact(-INF, INF, 10) not in ns_limited

    assert ns in ns
    assert ns_limited in ns
    assert ns not in ns_limited

    # test : ([s1, e1]:0)S_0([s2, e2]:w2) == [max(s1,s2),min(e1,e2),w2]
    # print(Nevada(Contact(2, 5, 0), Contact(3, 8, 2), Storage(0)))
    # print(Nevada(Contact(3, 5, 2)))
    assert Nevada(Contact(2, 5, 0), Contact(3, 8, 2), Storage(0)) == Nevada(Contact(3, 5, 2))

    # n = Nevada(Contact(3, 6, 0), Contact(3, 6, 2), Storage(0))
    # print(get_ascii_diagram(n, 1, 10, step=0.5))
    # print(n.get_boundary())

    # print(get_ascii_diagram(Contact(2, 5, 1), 1, 6, step=0.5))
    # n = Nevada(Contact(2, 5, 1))
    # print(get_ascii_diagram(n, 1, 6, step=0.5))
    # print(Nevada(Contact(2, 4, 2)))

    # points = Nevada(Contact(0, 1, 2)).get_boundary()
    # print(f"{points = }")

    # test : __mul__ and __rmul__
    
    n = Nevada(Contact.identity(), Contact(2, 5, 0), Storage())
    assert n == Storage() * Contact(2, 5, 0)
    assert n == ns * Contact(2, 5, 0)
    assert n == ns * n

    nn = Nevada(Contact(2, 5, 0), Contact.identity(), Storage())
    assert nn == Contact(2, 5, 0) * Storage() # rmul of storage
    assert nn == Contact(2, 5, 0) * ns # rmul of nevada
    assert nn == nn * ns

    # print(get_ascii_diagram(nn, 1, 6, step=0.5))
    # print(get_ascii_diagram(nn * ns, 1, 6, step=0.5))
    

class Product():
    
    def __init__(self, sequence: list[Nevada | Storage | Contact]):
        
        if len(sequence) == 0:
            raise ValueError(f"`sequence` must not be empty")
        
        self.sequence = sequence

        # keep track of simplified form; 
        # maybe pass through to avoid recalculating
        e: Nevada | Storage | Contact = Contact.identity()
        for s in sequence:
            e = e * s
        self.evaluated = e

        # possibly keep track of boundary as well

    def append(self, other: Product | Nevada | Storage | Contact) -> None:
        if isinstance(other, Product):
            self.sequence = self.sequence + other.sequence
            self.evaluated = self.evaluated * other.evaluated
        else:
            self.sequence.append(other)
            self.evaluated = self.evaluated * other

        return None

    def get_boundary(self) -> list[Point]:
        return self.evaluated.get_boundary()

    def __getitem__(self, index: Point) -> bool | float:
        return self.get_entry(index[0], index[1])

    def get_entry(self, i: float, j: float, flag: bool = False) -> bool | float:
        return self.evaluated.get_entry(i, j, flag)

    def __add__(self, other: Product | Nevada | Storage | Contact) -> Sum:
        # TODO : add containment logic

        return Sum([self, other])

    def __radd__(self, other: Nevada | Storage | Contact) -> Sum:
        return Sum([other, self])

    def __mul__(self, other: Product | Nevada | Storage | Contact) -> Product:
        if isinstance(other, Product):
            return Product(self.sequence + other.sequence)

        # if isinstance(other, Nevada) or isinstance(other, Storage) or \
        #         isinstance(other, Contact):
        return Product(self.sequence + [other])

    def __rmul__(self, other: Nevada | Storage | Contact) -> Product:

        return Product([other] + self.sequence)

    def __contains__(self, other: Product | Nevada | Storage | Contact) -> bool:
        # TODO : implement
        return False

        # return NotImplemented
    def __eq__(self, other) -> bool:
        if isinstance(other, Product):
            return self.evaluated == other.evaluated

        return NotImplemented

    def __format__(self, spec: str) -> str:
        if spec in ["e", "t"]:
            return f"{self.evaluated}"
        return str(self)

    def __str__(self) -> str:
        return "*".join([str(e) for e in self.sequence])

# `Product` class unit tests
if __name__ == "__main__":

    sequence: list[Nevada | Storage | Contact] = [
        Contact(0, 10, 5),
        Storage(),
        Storage(),
        Contact(3, 6, 2),
        Storage(),
        Contact(1, 8, 1),
        Contact(0, 8, 2)
    ]

    sequence_standard: list[Nevada | Storage | Contact] = [
        Contact(0, 1, 5),
        Storage(),
        Contact(3, 5, 5)
    ]

    sequence_nevada: list[Nevada | Storage | Contact] = [
        Nevada(Contact(0, 1, 5),Contact(3, 5, 5), Storage())
    ]

    assert Product(sequence_standard) == Product(sequence)
    assert Product(sequence_standard) == Product(sequence_nevada)

    sequence_standard = [
        Contact(0, 3, 0),
        Storage(1),
        Contact(3, 4, 0),
        Storage(0),
        Contact(2, 7, 0)
    ]
    # print(f"{Product(sequence):e}")

    sequence = [
        Contact(0, 11, 12),
        Storage(),
        Storage(),
        Contact(5, 20, 1),
        Contact(8, 90, 3),
        Storage(),
        Contact(4, INF, 6)
    ]

    sequence_standard = [
        Contact(0, 8, 12),
        Storage(),
        Contact(7, INF, 10)
    ]

    sequence_a: list[Nevada | Storage | Contact]= [
        Contact(0, 8, 12),
        Storage(),
        Contact(12, INF, 10)
    ]

    sequence_b: list[Nevada | Storage | Contact] = [
        Contact(0, 8, 22),
        Storage(),
        Contact(22, INF, 0)
    ]

    sequence_c: list[Nevada | Storage | Contact] = [
        Contact(0, 8, 0),
        Storage(),
        Contact(0, INF, 22)
    ]

    # print(f"{Product(sequence)} = {Product(sequence):e}")
    # print(f"{Product(sequence_standard)} = {Product(sequence_standard):e}")

    assert Product(sequence_standard) == Product(sequence)
    assert Product(sequence_standard) == Product(sequence_a)
    assert Product(sequence_standard) == Product(sequence_b)
    assert Product(sequence_standard) == Product(sequence_c)


class Sum():

    def __init__(self, 
        elements: list[Product | Nevada | Storage | Contact], 
        epsilon: float = 0
    ) -> None:

        self.elements = elements

    def __getitem__(self, index: Point) -> bool | float:
        return self.get_entry(index[0], index[1])

    def get_entry(self, i: float, j: float, flag: bool = False) -> bool | float:

        if flag:
            value_b: bool = False
            for e in self.elements:
                value_b = value_b or bool(e.get_entry(i, j, flag))
            return value_b

        value: float = 0
        for e in self.elements:
            value = max(value, e.get_entry(i, j))

        return value

    # doesn't make sense
    def get_boundary(self) -> list[Point]:
        points = []
        for e in self.elements:
            points += e.get_boundary()
        return list(set(points))

    def append(self, other: Sum | Product | Nevada | Storage | Contact) -> None:
        if isinstance(other, Sum):
            for e in other.elements:
                self.append(e)
        else:
            if other in self:
                self.elements.append(other)

        return None

    def __contains__(self, 
        other: Sum | Product | Nevada | Storage | Contact
    ) -> bool:
        if isinstance(other, Sum):
            for e in other.elements:
                if e not in self:
                    return False
            return True
        else:
            if other in self:
                return True

        return False

    # += should use append logic
    def __iadd__(self, 
        other: Sum | Product | Nevada | Storage | Contact
    ) -> Self:
        self.append(other)

        return self

    def __add__(self, other: Sum | Product | Nevada | Storage | Contact) -> Sum:
        if isinstance(other, Sum):
            return Sum(self.elements + other.elements)
        
        return Sum(self.elements + [other])

    def __radd__(self, other: Product | Nevada | Storage | Contact) -> Sum:
        return Sum([other] + self.elements)

    def __mul__(self, other: Sum | Product | Nevada | Storage | Contact) -> Sum:

        elements: list[Product | Nevada | Storage | Contact] = []

        if isinstance(other, Sum):
            pairs = [(i, j) for i in self.elements for j in other.elements]
            for i, j in pairs:
                elements.append(i * j)
        else:
            elements.append(other)

        return Sum(elements)

    def __format__(self, spec: str) -> str:
        if spec == "t":
            return " + ".join([f"{e:t}" for e in self.elements])
        return str(self)

    def __str__(self) -> str:
        return " + ".join([str(e) for e in self.elements])

if __name__ == "__main__":
    # p = Product([])
    # pp = Product([])

    a = Sum([Contact(1, 2, 5), Storage(2), Contact(0, 3, 4)])
    a = Sum([Contact(0, 3, 0) * Storage() * Contact(1, 4, 0)])
    assert a[0, 5] == 0
    assert a[1, 3] == 1

    # print(a.get_boundary())
    # print(f"{a + s}")
    # print(f"{s + a}")


# this is essentially another for of Product class
class ContactSequence():


    def __init__(self, 
        sequence: list[Nevada | Storage | Contact],
        summary: ContactSequenceSummary | None = None
    ) -> None:

        contact_sequence: list[Contact] = []
        storage_sequence: list[Storage] = []

        for i, e in enumerate(sequence):
            m, n = len(contact_sequence), len(storage_sequence)
            if isinstance(e, Contact):
                if m == n + 1:
                    contact_sequence[-1] *= e
                else:
                    contact_sequence.append(e)
            elif isinstance(e, Storage):
                if m == 0:
                    contact_sequence.append(Contact.identity())
                    storage_sequence.append(e)
                elif m == n:
                    product = storage_sequence[-1] * e
                    assert isinstance(product, Storage)

                    storage_sequence[-1] = product
                else:
                    storage_sequence.append(e)
            elif isinstance(e, Nevada):
                if m == n:
                    contact_sequence.append(e.left)
                    storage_sequence.append(e.storage)
                    contact_sequence.append(e.right)
                else:
                    contact_sequence[-1] *= e.left
                    storage_sequence.append(e.storage)
                    contact_sequence.append(e.right)
            else:
                raise ValueError(f"Invalid object type in `sequence` : {e}")

            # for i, a in enumerate(contact_sequence):
            #     print(f"\tcs[{i}] = {a}")
            # for j, b in enumerate(storage_sequence):
            #     print(f"\tss[{j}] = {b}")

        self.contact_sequence = contact_sequence
        self.storage_sequence = storage_sequence

        # if summary is not None:
        self.summary = summary

        return None

    def get_sequence(self) -> list[Nevada | Storage | Contact]:

        c, s = self.contact_sequence, self.storage_sequence

        interleaved: list[Nevada | Storage | Contact] = []
            # = [e for p in zip(c, s) for e in p] + [c[-1]]
        for p in zip(c, s):
            for e in p:
                interleaved.append(e)
        interleaved.append(c[-1])

        return interleaved

    def get_entry(self, i: float, j: float, flag:bool = False) -> bool | float:
        return self.summarize().get_entry(i, j, flag)

    def to_nevada(self) -> Nevada:
        return self.summarize().to_nevada()

    def get_boundary(self) -> list[Point]:
        return self.summarize().get_boundary()

    def append(self, other: ContactSequence | Nevada | Storage | Contact) -> None:

        if isinstance(other, ContactSequence):
            self.contact_sequence += other.contact_sequence
            self.storage_sequence += other.storage_sequence
            
            self.summary = self.summarize() * other.summarize()
        else:
            self.append(ContactSequence([other]))

        return None

    def get_summary(self) -> ContactSequenceSummary: # TODO : delete none
        # assuming self.sequence is of the form c S c S ... S c
        c, m = self.contact_sequence, len(self.contact_sequence)
        s, n = self.storage_sequence, len(self.storage_sequence)

        assert m == n + 1, "contact and storage sequences incorrect form"

        cumulant_delay = [sum(x.delay for x in c[:i + 1])
                                            for i in range(m)] + [0]
        cumulant_storage = [sum([y.capacity for y in s[:i + 1]])
                                            for i in range(n)] + [0]
        start_adjusted = [x.start - cumulant_delay[i - 1] 
                                            for i, x in enumerate(c)]
        end_adjusted = [x.end - cumulant_delay[i - 1] 
                                            for i, x in enumerate(c)]

        E = min(end_adjusted[:-1])
        epsilon = end_adjusted[-1]
        e = min(end_adjusted[0], end_adjusted[1]) # TODO : double check
        # print(f"{end_adjusted[0] = }, {end_adjusted[1] = }")
        tau = min([end - max(start_adjusted[:k + 1], default=0) 
            for k, end in enumerate(end_adjusted)])
        # omega = sum(cumulant_delay)
        omega = cumulant_delay[-2]
        # A = sum(cumulant_storage)
        A = cumulant_storage[-2]
        rho = min([end_adjusted[-1]] + \
            [end_adjusted[k] + cumulant_delay[n - 2] - cumulant_delay[k - 1]
                for k in range(n)])
        sigma = max([start_adjusted[i] - cumulant_storage[i - 1] 
                        for i in range(n)])
        S = max(start_adjusted)

        return ContactSequenceSummary(E, epsilon, e, tau, omega, A, rho, sigma, S)

    def summarize(self) -> ContactSequenceSummary:
        if self.summary is not None:
            return self.summary
        self.summary = self.get_summary()

        return self.summary

    def __mul__(self, 
        other: ContactSequence | Nevada | Storage | Contact
    ) -> ContactSequence:

        if isinstance(other, ContactSequence):
            return ContactSequence(
                self.get_sequence() + other.get_sequence(), 
                self.summarize() * other.summarize()
            )
        else:
            return self * ContactSequence([other])

    def __eq__(self, other) -> bool:

        if isinstance(other, ContactSequence):
            return self.summarize() == other.summarize()
            # return self.summarize().to_nevada() == other.summarize().to_nevada()
        return False

    def __str__(self) -> str:
        # c, s = self.contact_sequence, self.storage_sequence
        # interleaved = [e for p in zip(c, s) for e in p] + [c[-1]]
        sequence = self.get_sequence()
        return "*".join([f"{e}" for e in sequence])

class ContactSequenceSummary():

    def __init__(self,
        E: float,
        epsilon: float,
        e: float,
        tau: float,
        omega: float,
        A: float,
        rho: float,
        sigma: float,
        S: float
    ) -> None:

        self.E = E # adjusted end times not including last
        self.epsilon = epsilon # final adjusted end time
        self.e = e # first contact end time
        self.tau = tau # maximum throughput
        self.omega = omega # total delay
        self.A = A # total cumulant storage
        self.rho = rho # storage requirement
        self.sigma = sigma # first (adjusted) start time
        self.S = S # maximum adjusted start time

        return None

    def contains_point(self, i: float, j: float) -> bool:

        condition_a = self.sigma <= i and i <= min(self.E, self.epsilon)
        if not condition_a:
            return False

        condition_b = self.S + self.omega <= j and \
            j <= min(self.e + self.A, self.rho) + self.omega
        if not condition_b:
            return False

        condition_c = i + self.omega <= j and j <= i + self.omega + self.A
        if not condition_c:
            return False

        return True

    def get_entry(self, i: float, j: float, flag: bool = False) -> bool | float:

        contained = self.contains_point(i, j)

        if flag:
            return contained

        value = min(self.E - i, self.tau, self.rho + self.omega - j)
        return value if contained else 0

    def get_boundary(self) -> list[Point]:
        return self.to_nevada().get_boundary()

    def __mul__(self, other: ContactSequenceSummary) -> ContactSequenceSummary:

        return ContactSequenceSummary(
            min(self.E, self.epsilon, other.E - self.omega),
            other.epsilon - self.omega,
            self.epsilon,
            min(self.tau, other.tau, 
                min(other.E, other.epsilon) - self.omega - self.S),
            self.omega + other.omega,
            self.A + other.A,
            min(self.rho, other.e - self.A, other.rho - self.A - self.omega, 
                other.epsilon - self.omega),
            max(self.sigma, other.sigma - self.A - self.omega),
            max(self.S, other.S - self.omega)
        )

    def to_nevada(self) -> Nevada:
        return Nevada(
            Contact(self.sigma, min(self.E, self.epsilon), 0), 
            Contact(self.S, min(self.e + self.A, self.rho), self.rho), 
            Storage(self.A)
        )

    def __eq__(self, other) -> bool:
        if isinstance(other, ContactSequenceSummary):
            return self.__dict__ == other.__dict__
        return False
    
    def __format__(self, spec: str) -> str:
        if spec == "e":
            return f"{self.to_nevada()}"
        return str(self)

    def __str__(self) -> str:
        return " | ".join([f"{key} : {value}" 
            for key, value in self.__dict__.items()])

# `ContactSequence` class unit tests
if __name__ == "__main__":

    sequence = [
        Contact(0, 11, 12),
        Storage(),
        Storage(),
        Contact(5, 20, 1),
        Contact(8, 90, 3),
        Storage(),
        Contact(4, INF, 6)
    ]

    sequence_standard = [Contact(0, 8, 12), Storage(), Contact(7, INF, 10)]

    sequence_nevada_a: list[Nevada | Storage | Contact] = [
        Nevada(Contact(0, 8, 12), Contact(12, INF, 10), Storage())
    ]

    sequence_nevada_b: list[Nevada | Storage | Contact] = [
        Nevada(Contact(0, 8, 22), Contact(22, INF, 0), Storage())
    ]

    sequence_nevada_c: list[Nevada | Storage | Contact] = [
        Nevada(Contact(0, 8, 0), Contact(0, INF, 22), Storage())
    ]

    cs = ContactSequence(sequence)
    cs_std = ContactSequence(sequence_standard)
    csn_a = ContactSequence(sequence_nevada_a)
    csn_b = ContactSequence(sequence_nevada_b)
    csn_c = ContactSequence(sequence_nevada_c)

    # print(f"{cs}")
    # print(f"{cs.summarize()}")
    # print(f"{cs.to_nevada()}")
    # print(f"{cs_std}")
    # print(f"{cs_std.summarize()}")
    # print(f"{cs_std.to_nevada()}")
    # print(f"{csn}")


    assert cs_std.to_nevada() == cs.to_nevada()
    assert cs_std == cs
    assert cs_std == csn_a
    assert cs_std == csn_b
    assert cs_std == csn_c

# TODO : maybe rename to `SemiringMatrix`
Element = Sum | Product | Nevada | Storage | Contact
class Matrix():
    
    def __init__(self, m: int, n: int, array: list[list[Element]]) -> None:

        self.dim_row = m
        self.dim_col = n
        self.array = array

        return None

    def __getitem__(self, index: tuple[int, int]) -> Element:
        return self.array[index[0]][index[1]]
    
    def __setitem__(self, index: tuple[int, int], value: Element) -> None:
        self.array[index[0]][index[1]] = value

    def __add__(self, other: Matrix) -> Matrix:
        if self.dim_row != other.dim_row or self.dim_col != other.dim_col:
            raise ValueError("Dimension mismatch in addition")

        array = Matrix.empty_array(self.dim_row, self.dim_col)
        for i, j in Matrix.get_indices(self.dim_row, self.dim_col):
            array[i][j] = self[i, j] + other[i, j]

        return Matrix(self.dim_row, self.dim_col, array)

    def __mul__(self, other: Matrix) -> Matrix:
        if self.dim_col != other.dim_row:
            raise ValueError("Dimension mismatch in multiplication")

        array = Matrix.empty_array(self.dim_row, other.dim_col)
        for i, j in Matrix.get_indices(self.dim_row, other.dim_col):
            for k in range(self.dim_col):
                # __iadd__ appends instead of creating new Sum object on LHS
                array[i][j] += self[i, k] + other[k, j]

        return Matrix(self.dim_row, self.dim_col, array)

    @staticmethod
    def empty_array(dim_row: int, dim_col: int) -> list[list[P.Interval]]:
        return [[Sum([]) for c in range(dim_col)] for r in range(dim_row)]

    @staticmethod
    def get_indices(rows: int, columns: int) -> list[tuple[int, int]]:
        return [(i, j) for i in range(rows) for j in range(columns)]

# help(Contact)

exit()

# TODO : delete following
# `ContactSequence` class unit tests
if __name__ == "__main__":

    sequence = [
        Contact(0, 10, 5),
        Storage(),
        Storage(),
        Contact(3, 6, 2),
        Storage(),
        Contact(1, 8, 1),
        Contact(0, 8, 2)
    ]

    sequence_standard = [
        Contact(0, 1, 5),
        Storage(),
        Contact(3, 5, 5)
    ]

    sequence_nevada = [
        Nevada(Contact(0, 1, 5),Contact(3, 5, 5), Storage())
    ]

    print(ContactSequence(sequence_standard))
    print(ContactSequence(sequence))
    print(ContactSequence(sequence_standard).summarize())
    print(ContactSequence(sequence).summarize())
    assert ContactSequence(sequence_standard) == ContactSequence(sequence)
    assert ContactSequence(sequence_standard) == ContactSequence(sequence_nevada)

    sequence_standard = [
        Contact(0, 3, 0),
        Storage(1),
        Contact(3, 4, 0),
        Storage(0),
        Contact(2, 7, 0)
    ]
    # print(f"{ContactSequence(sequence):e}")

    sequence = [
        Contact(0, 11, 12),
        Storage(),
        Storage(),
        Contact(5, 20, 1),
        Contact(8, 90, 3),
        Storage(),
        Contact(4, INF, 6)
    ]

    sequence_standard = [
        Contact(0, 8, 12),
        Storage(),
        Contact(7, INF, 10)
    ]

    sequence_a = [
        Contact(0, 8, 12),
        Storage(),
        Contact(12, INF, 10)
    ]

    sequence_b = [
        Contact(0, 8, 22),
        Storage(),
        Contact(22, INF, 0)
    ]

    sequence_c = [
        Contact(0, 8, 0),
        Storage(),
        Contact(0, INF, 22)
    ]

    # print(f"{ContactSequence(sequence)} = {ContactSequence(sequence):e}")
    # print(f"{ContactSequence(sequence_standard)} = {ContactSequence(sequence_standard):e}")

    assert ContactSequence(sequence_standard) == ContactSequence(sequence)
    assert ContactSequence(sequence_standard) == ContactSequence(sequence_a)
    assert ContactSequence(sequence_standard) == ContactSequence(sequence_b)
    assert ContactSequence(sequence_standard) == ContactSequence(sequence_c)