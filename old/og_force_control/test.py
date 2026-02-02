from potential_field import PotentialField, Point

class PotentialPoint(Point):
    """ A point in the potential field that has a value based on the distance from the origin """
    def __init__(self, loc: tuple[float, float, float]):
        f = lambda x: (x[0]**2 + x[1]**2 + x[2]**2)**0.5
        super().__init__(loc, f)

field = PotentialField((1, 1, 1), point_type=PotentialPoint)
point = PotentialPoint((0.49, 0.49, 0.49))
print(field())