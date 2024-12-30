from ortools.sat.python import cp_model
from typing import List, Dict
from dataclasses import dataclass
from datetime import timedelta, datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import networkx as nx
from collections import defaultdict
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
%matplotlib inline
import numpy as np

@dataclass(frozen=True)
class Course:
    name: str
    lecturer: str
    duration: int
    group: str
    students: int
    session_id: int

@dataclass
class ClassRoom:
    name: str
    capacity: int

class TimetableCSP:
    def __init__(self):
        self.model = cp_model.CpModel()
        self.variables = {}
        self.course_data = []

        # Define data
        self.classrooms = [
            ClassRoom("7609", 80),
            ClassRoom("7610", 80),
            ClassRoom("Multimedia", 80),
            ClassRoom("GKUT", 80),
            ClassRoom("Zoom", 160),
        ]

        self.days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        self.day_start_times = {
            "Monday": "13:00",
            "Tuesday": "13:00",
            "Wednesday": "09:00",
            "Thursday": "13:00",
            "Friday": "14:00",
        }
        self.day_end_times = {
            "Monday": "17:00",
            "Tuesday": "18:00",
            "Wednesday": "18:00",
            "Thursday": "18:00",
            "Friday": "16:00",
        }

        self.groups = {
            "M1": [
                Course("IF1230", "Achmad Imam Kistijantoro, S.T, M.Sc., Ph.D.", 2, "M1", 80, 1),
                Course("IF1230", "Achmad Imam Kistijantoro, S.T, M.Sc., Ph.D.", 1, "M1", 80, 2),
                Course("IF2150", "Dr. tech. Wikan Danar Sunindyo, S.T, M.Sc.", 2, "M1", 80, 1),
                Course("IF2150", "Dr. tech. Wikan Danar Sunindyo, S.T, M.Sc.", 1, "M1", 80, 2),
                Course("IF2123", "Dr. Ir. Rinaldi, M.T.", 2, "M1", 80, 1),
                Course("IF2123", "Dr. Ir. Rinaldi, M.T.", 1, "M1", 80, 2),
                Course("IF2110", "Dr. Yani Widyani, S.T, M.T.", 2, "M1", 80, 1),
                Course("IF2110", "Dr. Yani Widyani, S.T, M.T.", 2, "M1", 80, 2),
                Course("IF1220", "Ir. Rila Mandala, M.Eng., Ph.D.", 1, "M1", 80, 1),
                Course("IF1220", "Ir. Rila Mandala, M.Eng., Ph.D.", 2, "M1", 80, 2),
                Course("KU4078", "Ridwan Fauzi, S.Pd., MH.", 2, "M1", 80, 1),
                Course("KU2071", "Prof. Ir. Dicky Rezady Munaf, M.S, M.SCE, Ph.D.", 2, "M1", 80, 1),
                Course("IF1221", "Ir. Rila Mandala, M.Eng., Ph.D.", 2, "M1", 80, 1),
            ],
            "M2": [
                Course("IF1230", "Asisten Dosen", 2, "M2", 80, 1),
                Course("IF1230", "Asisten Dosen", 1, "M2", 80, 2),
                Course("IF2150", "Dr. Yani Widyani, S.T, M.T.", 2, "M2", 80, 1),
                Course("IF2150", "Dr. Yani Widyani, S.T, M.T.", 1, "M2", 80, 2),
                Course("IF2123", "Ir. Rila Mandala, M.Eng., Ph.D.", 2, "M2", 80, 1),
                Course("IF2123", "Ir. Rila Mandala, M.Eng., Ph.D.", 1, "M2", 80, 2),
                Course("IF2110", "Dr. Phil. Eng. Hari Purnama, S.Si., M.Si.", 2, "M2", 80, 1),
                Course("IF2110", "Dr. Phil. Eng. Hari Purnama, S.Si., M.Si.", 2, "M2", 80, 2),
                Course("IF1220", "Dr. Ir. Rinaldi, M.T.", 1, "M2", 80, 1),
                Course("IF1220", "Dr. Ir. Rinaldi, M.T.", 2, "M2", 80, 2),
                Course("KU4078", "Ridwan Fauzi, S.Pd., MH.", 2, "M2", 80, 1),
                Course("KU2071", "Ir. Siti Kusumawati Azhari, S.H, M.T.", 2, "M2", 80, 1),
                Course("IF1221", "Dr. Judhi Santoso, M.Sc.", 2, "M2", 80, 1),
            ],
        }

        self.intervals = self.generate_intervals()

    def generate_intervals(self):
        intervals = {}
        for day in self.days:
            start = datetime.strptime(self.day_start_times[day], "%H:%M")
            end = datetime.strptime(self.day_end_times[day], "%H:%M")
            slots = []
            while start + timedelta(hours=1) <= end:
                slot_end = start + timedelta(hours=1)
                slots.append((start.strftime("%H:%M"), slot_end.strftime("%H:%M")))
                start = slot_end
            intervals[day] = slots
        return intervals

    def create_variables(self):
        # Original variables for individual sessions
        for group, courses in self.groups.items():
            for course in courses:
                for day, slots in self.intervals.items():
                    for slot_idx in range(len(slots) - course.duration + 1):
                        for room_idx, room in enumerate(self.classrooms):
                            var_name = f"{course.group}_{course.name}_{course.session_id}_{day}_{slot_idx}_{room_idx}"
                            self.variables[var_name] = self.model.NewBoolVar(var_name)
                            self.course_data.append((var_name, course, day, slot_idx, room))

        # Additional variables for combined sessions
        m1_courses = {(c.name, c.lecturer, c.session_id, c.duration): c for c in self.groups["M1"]}
        m2_courses = {(c.name, c.lecturer, c.session_id, c.duration): c for c in self.groups["M2"]}

        # Find matching courses between M1 and M2
        matching_courses = set(m1_courses.keys()) & set(m2_courses.keys())

        # Create variables for combined sessions
        for course_key in matching_courses:
            course_m1 = m1_courses[course_key]
            course_m2 = m2_courses[course_key]
            total_students = course_m1.students + course_m2.students

            for day, slots in self.intervals.items():
                for slot_idx in range(len(slots) - course_m1.duration + 1):
                    # Only create combined session for rooms with enough capacity
                    for room_idx, room in enumerate(self.classrooms):
                        if room.capacity >= total_students:
                            combined_var_name = f"COMBINED_M1M2_{course_m1.name}_{course_m1.session_id}_{day}_{slot_idx}_{room_idx}"
                            self.variables[combined_var_name] = self.model.NewBoolVar(combined_var_name)

                            # Create a special course object for the combined session
                            combined_course = Course(
                                name=course_m1.name,
                                lecturer=course_m1.lecturer,
                                duration=course_m1.duration,
                                group="M1+M2",
                                students=total_students,
                                session_id=course_m1.session_id
                            )
                            self.course_data.append((combined_var_name, combined_course, day, slot_idx, room))

    def add_constraints(self):
        # Helper function to get course key
        def get_course_key(course):
            return (course.name, course.lecturer, course.session_id)

        # Create mapping of matching courses
        m1_courses = {get_course_key(c): c for c in self.groups["M1"]}
        m2_courses = {get_course_key(c): c for c in self.groups["M2"]}
        matching_courses = set(m1_courses.keys()) & set(m2_courses.keys())

        # Modified constraint: Each course must be assigned exactly once (either individual or combined)
        for course_data in set(cd for _, cd, _, _, _ in self.course_data):
            course_key = get_course_key(course_data)
            if course_key in matching_courses and course_data.group in ["M1", "M2"]:
                relevant_vars = [
                    vn for vn, cd, _, _, _ in self.course_data 
                    if (cd.group in [course_data.group, "M1+M2"] and 
                        cd.name == course_data.name and 
                        cd.session_id == course_data.session_id)
                ]
                self.model.Add(sum(self.variables[vn] for vn in relevant_vars) == 1)
            elif course_data.group != "M1+M2":
                relevant_vars = [
                    vn for vn, cd, _, _, _ in self.course_data 
                    if cd == course_data
                ]
                self.model.Add(sum(self.variables[vn] for vn in relevant_vars) == 1)

        # Prevent overlapping sessions for the same room
        for day, slots in self.intervals.items():
            for slot_idx in range(len(slots)):
                for room in self.classrooms:
                    overlapping_slots = [
                        vn for vn, cd, d, s, r in self.course_data
                        if d == day and r.name == room.name and
                        any(s + offset == slot_idx for offset in range(cd.duration))
                    ]
                    self.model.Add(sum(self.variables[vn] for vn in overlapping_slots) <= 1)

        # Prevent overlapping sessions for the same group (modified to handle combined sessions)
        for day, slots in self.intervals.items():
            for slot_idx in range(len(slots)):
                for group in ["M1", "M2"]:
                    overlapping_slots = [
                        vn for vn, cd, d, s, r in self.course_data
                        if (cd.group == group or cd.group == "M1+M2") and d == day and
                        any(s + offset == slot_idx for offset in range(cd.duration))
                    ]
                    self.model.Add(sum(self.variables[vn] for vn in overlapping_slots) <= 1)

        # NEW: Prevent same lecturer from teaching different courses at the same time
        for day, slots in self.intervals.items():
            for slot_idx in range(len(slots)):
                # Get all unique lecturers
                all_lecturers = set(cd.lecturer for _, cd, _, _, _ in self.course_data)

                for lecturer in all_lecturers:
                    overlapping_slots = [
                        vn for vn, cd, d, s, r in self.course_data
                        if cd.lecturer == lecturer and d == day and
                        any(s + offset == slot_idx for offset in range(cd.duration))
                    ]
                    # Ensure lecturer is only teaching one course at a time
                    self.model.Add(sum(self.variables[vn] for vn in overlapping_slots) <= 1)

        # Enforce room capacity constraints
        for day, slots in self.intervals.items():
            for slot_idx in range(len(slots)):
                for room in self.classrooms:
                    self.model.Add(
                        sum(
                            self.variables[vn] * cd.students
                            for vn, cd, d, s, r in self.course_data
                            if d == day and s == slot_idx and r.name == room.name
                        )
                        <= room.capacity
                    )

        # Restrict same course of the same group from being scheduled on the same day
        for group in ["M1", "M2"]:
            for course in self.groups[group]:
                for day in self.days:
                    same_course_sessions = [
                        vn for vn, cd, d, s, r in self.course_data
                        if ((cd.group == group or cd.group == "M1+M2") and 
                            cd.name == course.name and d == day)
                    ]
                    self.model.Add(
                        sum(self.variables[vn] for vn in same_course_sessions) <= 1
                    )
                    
    def visualize_graphs(self, schedule_by_day=None):
        """
        Visualize both the initial conflict graph and the final scheduled graph.
        If schedule_by_day is None, only show the initial conflict graph.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        # Create initial conflict graph (before scheduling)
        G_initial = nx.Graph()
        node_labels_initial = {}

        # Add nodes for all possible courses
        for group, courses in self.groups.items():
            for course in courses:
                node = f"{course.name}_{group}_{course.session_id}"
                G_initial.add_node(node)
                node_labels_initial[node] = f"{course.name}\n({group}, S{course.session_id})"

                # Add edges for potential conflicts within the same group
                for other_course in courses:
                    if course != other_course:
                        other_node = f"{other_course.name}_{group}_{other_course.session_id}"
                        G_initial.add_edge(node, other_node)

                # Add edges for same course different groups
                for other_group, other_courses in self.groups.items():
                    if group != other_group:
                        for other_course in other_courses:
                            if course.name == other_course.name and course.lecturer == other_course.lecturer:
                                other_node = f"{other_course.name}_{other_group}_{other_course.session_id}"
                                G_initial.add_edge(node, other_node)

        # Draw initial graph
        pos_initial = nx.spring_layout(G_initial, k=1, iterations=50, seed=42)

        # Draw edges
        nx.draw_networkx_edges(G_initial, pos_initial, 
                              edge_color='gray',
                              alpha=0.3,
                              width=1,
                              ax=ax1)

        # Draw nodes
        nx.draw_networkx_nodes(G_initial, pos_initial,
                              node_color='lightblue',
                              node_size=2000,
                              ax=ax1)

        # Draw labels
        nx.draw_networkx_labels(G_initial, pos_initial,
                               labels=node_labels_initial,
                               font_size=8,
                               font_weight='bold',
                               ax=ax1)

        ax1.set_title("Initial Conflict Graph\n(Before Scheduling)")

        # Draw final scheduled graph if schedule exists
        if schedule_by_day:
            G_final = nx.Graph()
            node_labels_final = {}

            # Create nodes for scheduled sessions
            for day, sessions in schedule_by_day.items():
                for session in sessions:
                    node = f"{session['course']}_{session['group']}_{session['session_id']}"
                    G_final.add_node(node, 
                                   time=f"{day} {session['start_time']}-{session['end_time']}",
                                   day=day,
                                   start_time=session['start_time'],
                                   group=session['group'],
                                   lecturer=session['lecturer'],
                                   room=session['room'])
                    node_labels_final[node] = f"{session['course']}\n({session['group']}, S{session['session_id']})"

            # Add edges for actual conflicts in schedule
            for day, sessions in schedule_by_day.items():
                day_nodes = [
                    (f"{s['course']}_{s['group']}_{s['session_id']}", s)
                    for s in sessions
                ]

                for i, (node1, session1) in enumerate(day_nodes):
                    for node2, session2 in day_nodes[i+1:]:
                        if (
                            session1['start_time'] == session2['start_time'] and
                            (session1['lecturer'] == session2['lecturer'] or
                             session1['group'] == session2['group'] or
                             session1['room'] == session2['room'])
                        ):
                            G_final.add_edge(node1, node2)

            # Apply coloring to final graph
            coloring = nx.coloring.greedy_color(G_final, strategy='largest_first')
            unique_colors = sorted(set(coloring.values()))
            color_map = plt.cm.rainbow(np.linspace(0, 1, len(unique_colors)))
            node_colors = [color_map[coloring[node]] for node in G_final.nodes()]

            # Draw final graph
            pos_final = nx.spring_layout(G_final, k=1, iterations=50, seed=42)

            nx.draw_networkx_edges(G_final, pos_final, 
                                 edge_color='gray',
                                 alpha=0.5,
                                 width=1,
                                 ax=ax2)

            nx.draw_networkx_nodes(G_final, pos_final,
                                 node_color=node_colors,
                                 node_size=2000,
                                 ax=ax2)

            nx.draw_networkx_labels(G_final, pos_final,
                                  labels=node_labels_final,
                                  font_size=8,
                                  font_weight='bold',
                                  ax=ax2)

            ax2.set_title("Final Scheduled Graph\n(After Scheduling)")

            # Add colorbar for time slots
            sm = ScalarMappable(cmap=plt.cm.rainbow, 
                               norm=Normalize(vmin=0, vmax=len(unique_colors)-1))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax2, orientation='vertical', fraction=0.046, pad=0.04)
            cbar.set_label("Time Slots", rotation=270, labelpad=15)

        plt.tight_layout()
        plt.show()
    
    def solve(self):
        solver = cp_model.CpSolver()
        solver.parameters.log_search_progress = False  # Disable verbose logging
        solver.parameters.num_search_workers = 8  # Multi-threading
        status = solver.Solve(self.model)

        if status in [cp_model.FEASIBLE, cp_model.OPTIMAL]:
            # Group schedule by day
            schedule_by_day = {day: [] for day in self.days}

            for vn, course, day, slot_idx, room in self.course_data:
                if solver.Value(self.variables[vn]) == 1:
                    slot = self.intervals[day][slot_idx:slot_idx + course.duration]
                    start_time = slot[0][0]
                    end_time = slot[-1][1]
                    schedule_by_day[day].append({
                        "start_time": start_time,
                        "end_time": end_time,
                        "course": course.name,
                        "lecturer": course.lecturer,
                        "group": course.group,
                        "room": room.name,
                        "duration": course.duration,
                        "session_id": course.session_id
                    })

            # Sort each day's schedule by start time
            for day in schedule_by_day:
                schedule_by_day[day].sort(key=lambda x: x["start_time"])
            
            self.visualize_graphs(schedule_by_day)
            
            # Print the formatted schedule
            print("\nFinal Schedule:")
            print("=" * 90)
            for day, sessions in schedule_by_day.items():
                print(f"\n{day}:")
                print("-" * 90)
                if not sessions:
                    print("No classes scheduled")
                    continue
                for session in sessions:
                    print(
                        f"{session['start_time']}-{session['end_time']} | {session['course']:6} | "
                        f"{session['lecturer']:47} | {session['group']:2} | "
                        f"Room {session['room']:10} | {session['duration']}h | S{session['session_id']}"
                    )
        else:
            print("No solution found.")


    def run(self):
        self.create_variables()
        self.add_constraints()
        self.solve()


if __name__ == "__main__":
    optimizer = TimetableCSP()
    optimizer.run()