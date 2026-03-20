from kg_builder.kg_constructor import KGConstructor
from kg_builder.node_models import (
    CityNode, DistrictNode, AmenityNode, RoadSegmentNode, TransportModeNode
)
from kg_builder.relationship_models import InDistrict


def seed_all():
    kg = KGConstructor()

    # Districts
    districts = [
        DistrictNode("Colombo", "Western"),
        DistrictNode("Gampaha", "Western"),
        DistrictNode("Kalutara", "Western"),
    ]
    for d in districts:
        kg.create_district(d)

    # Cities — Phase 1 focus: Western Province
    cities = [
        CityNode("Piliyandala", "Colombo", "Western", 6.8011, 79.9169),
        CityNode("Maharagama", "Colombo", "Western", 6.8483, 79.9260),
        CityNode("Homagama", "Colombo", "Western", 6.8424, 80.0023),
        CityNode("Colombo", "Colombo", "Western", 6.9271, 79.8612),
        CityNode("Moratuwa", "Colombo", "Western", 6.7728, 79.8820),
        CityNode("Dehiwala", "Colombo", "Western", 6.8516, 79.8715),
    ]
    for c in cities:
        kg.create_city(c)
        kg.link_city_to_district(
            InDistrict(city_name=c.name, district_name=c.district)
        )

    # Amenities — full taxonomy
    amenity_map = {
        "Connectivity": ["Free WiFi", "Business Center", "Conference Room"],
        "Recreation": ["Swimming Pool", "Fitness Center", "Spa", "Garden"],
        "Dining": ["Restaurant", "Bar", "Room Service", "Breakfast Included"],
        "Transport": ["Airport Shuttle", "Free Parking", "Taxi Service"],
        "Accessibility": ["Wheelchair Access", "Elevator", "Ground Floor Rooms"],
        "Family": ["Family Rooms", "Children Play Area", "Baby Cot"],
        "Comfort": ["Air Conditioning", "Hot Water", "24h Front Desk"],
        "Safety": ["CCTV", "Security Guard", "Safe Deposit Box"],
    }
    for category, names in amenity_map.items():
        for name in names:
            kg.create_amenity(AmenityNode(name=name, category=category))

    # Road segments around Piliyandala
    roads = [
        RoadSegmentNode("Highlevel Road", "Highway", "Asphalt", "Good", 15.0),
        RoadSegmentNode("Piliyandala-Kesbewa Rd", "Main Road", "Asphalt", "Moderate", 10.0),
        RoadSegmentNode("Colombo-Galle Highway", "Highway", "Asphalt", "Good", 20.0),
        RoadSegmentNode("Homagama Road", "Main Road", "Asphalt", "Good", 12.0),
        RoadSegmentNode("Kesbewa Side Road", "Side Road", "Concrete", "Moderate", 8.0),
    ]
    for road in roads:
        with kg.driver.session() as s:
            s.run(
                """
                MERGE (r:RoadSegment {name: $name})
                SET r.road_type           = $road_type,
                    r.surface             = $surface,
                    r.condition           = $condition,
                    r.avg_travel_time_min = $avg_travel_time_min,
                    r.last_checked        = $last_checked
                """,
                road.__dict__,
            )

    # Transport modes
    transports = [
        TransportModeNode("Bus", 0.3, 0.5, "High"),
        TransportModeNode("Tuk-tuk", 0.6, 0.7, "Medium"),
        TransportModeNode("Taxi", 1.0, 0.9, "High"),
        TransportModeNode("Train", 0.2, 0.6, "Medium"),
        TransportModeNode("Walk", 0.0, 0.2, "High"),
    ]
    for t in transports:
        with kg.driver.session() as s:
            s.run(
                """
                MERGE (t:TransportMode {name: $name})
                SET t.cost_factor  = $cost_factor,
                    t.speed_factor = $speed_factor,
                    t.reliability  = $reliability
                """,
                t.__dict__,
            )

    kg.close()
    print("Seed data loaded: Districts, Cities, Amenities, Roads, Transport.")


if __name__ == "__main__":
    seed_all()
