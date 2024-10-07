"""
This contains classes and methods related to the 
Storage-Limited Store-and-Forward Nevada Semiring.
"""

from __future__ import annotations

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
        return P.closed(self.start, self.end)

    def contains_point(self, i: float, j: float) -> bool:

        if self.start <= i and i <= self.end and j == i + self.delay:
            return True

        return False

    def __getitem__(self, index: Point) -> bool | float:
        # NOTE : this assumes you want the max-min [0, infty] semiring
        return self.get_entry(index[0], index[1])

    def get_entry(self, i: float, j: float, flag: bool = False) -> bool | float:

        contained = self.contains_point(i, j)

        if flag:
            return contained

        return self.end - i if contained else 0

        # if self.contains_point(i, j):
        #     return True if self.boolean else self.end - i
        # return False if self.boolean else 0

    def get_boundary(self) -> list[Point]:
        points = [
            (self.start, self.start + self.delay),
            (self.end, self.end + self.delay)
        ]

        return list(set(points))

    def __eq__(self, other) -> bool:

        if isinstance(other, Contact):
            return self.delay == other.delay and \
                self.get_interval() == other.get_interval()

        return False

    def __contains__(self, other: Contact | Point) -> bool:

        if isinstance(other, tuple) and isinstance_point(other):
            return self.contains_point(*other)

        if isinstance(other, Contact):

            if self.delay != other.delay or \
                other.get_interval() not in self.get_interval():
                return False
            return True

        return False

    def __mul__(self, other: Contact) -> Contact:
        if not isinstance(other, Contact):
            return NotImplemented
            
        start = max(self.start, other.start - self.delay)
        end = min(self.end, other.end - self.delay)
        delay = self.delay + other.delay

        if end - start < 0:
            start, end, delay = 0, 0, 0

        return Contact(start, end, delay)

    def __format__(self, spec: str) -> str:
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

    def __contains__(self, other: Storage | Contact | Point) -> bool:

        if isinstance(other, tuple) and isinstance_point(other):
            return self.contains_point(*other)

        if isinstance(other, Contact):
            return self.contains_contact(other)

        if isinstance(other, Storage):
            return self.capacity >= other.capacity

        return False

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

    def __mul__(self, other: Contact | Storage | Nevada) -> Nevada:
                
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
    
    def __init__(self, 
        sequence: list[Nevada | Storage | Contact],
        summary: ContactSequenceSummary | None = None):
        
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

    def __mul__(self, other: Product | Nevada | Storage | Contact) -> Product:
        if isinstance(other, Product):
            return Product(self.sequence + other.sequence)

        # if isinstance(other, Nevada) or isinstance(other, Storage) or \
        #         isinstance(other, Contact):
        return Product(self.sequence + [other])

    def __rmul__(self, other: Nevada | Storage | Contact) -> Product:

        return Product([other] + self.sequence)

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

    size = 25
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

    def get_boundary(self) -> list[Point]:
        points = []
        for e in self.elements:
            points += e.get_boundary()
        return list(set(points))

    def append(self, other: Sum | Product | Nevada | Storage | Contact) -> None:
        # possibly check if each summand is already contained in some existing
        #   summand
        return None

    def __add__(self, other: Sum | Product | Nevada | Storage | Contact) -> Sum:
        if isinstance(other, Sum):
            return Sum(self.elements + other.elements)
        
        return Sum(self.elements + [other])

    # TODO : __radd__

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
    print(f"{a}")

class ContactSequence():


    def __init__(self, sequence: list[Nevada | Storage | Contact]) -> None:

        # self.sequence = sequence
        self.contact_sequence: list[Contact] = []
        self.storage_sequence: list[Storage] = []

        # TODO : prepare sequence
        # for e in sequence:
        #     if isinstance(e, Contact):

        # too much for an init
        n = len(self.contact_sequence)
        
        start_adjusted: list[float] = []
        end_adjusted: list[float] = []
        delay_cumulant: list[float] = []
        storage_cumulant: list[float] = []

        E = min(end_adjusted[:-1])
        epsilon = end_adjusted[-1]
        e = end_adjusted[0]
        tau = min([end - max(start_adjusted[:k + 1], default=0) 
            for k, end in enumerate(end_adjusted)])
        omega = sum(delay_cumulant)
        A = sum(storage_cumulant)
        rho = min([end_adjusted[-1]] + [end_adjusted[k] + storage_cumulant[n - 1] - storage_cumulant[k - 1] for k in range(n)])
        sigma = max([start_adjusted[i] - storage_cumulant[i - 1] for i in range(n)])
        S = max(start_adjusted)


        self.summary = ContactSequenceSummary(E, epsilon, e, tau, omega, A, rho, sigma, S)

        return None

    def append(self, other: ContactSequence | Nevada | Storage | Contact) -> None:

        if isinstance(other, ContactSequence):
            self.contact_sequence += other.contact_sequence
            self.storage_sequence += other.storage_sequence
            
            if self.summary == None:
                self.summary = self.get_summary()
            self.summary = self.summary * other.summary



        return None

    def get_summary(self) -> ContactSequenceSummary: # TODO : delete none
        # assuming self.sequence is of the form c S c S ... S c
        c, n = self.contact_sequence, len(self.contact_sequence)
        s = self.storage_sequence

        # TODO : double check all of this
        start_adjusted = [c[0].start]
        end_adjusted = [c[0].end]
        delay_cumulant = [c[0].delay]
        storage_cumulant = [0]

        for k in range(1, n):
            delay_k = c[k].delay + delay_cumulant[-1]
            start_k = c[k].start - delay_k
            # delay_cumulant.append(c[k].delay + delay_cumulant[-1])
            # start_adjusted.append()
        

        return ContactSequenceSummary(0, 0, 0, 0, 0, 0, 0, 0, 0)

    def summarize(self) -> ContactSequenceSummary:
        if self.summarize is not None:
            return self.summary
        self.summary = self.get_summary()

        return self.summary

    # def __getattr__(self, item: str) -> float:
    #     if item == "summary":
    #         if "summary" not in self.__dict__:
    #             self.__dict__["summary"] = self.get_summary()

    #     message = f"{item} is not a valid attribute of ContactSequence"
    #     raise AttributeError(message)


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

    def __str__(self) -> str:
        return f"{self.to_nevada():t}"