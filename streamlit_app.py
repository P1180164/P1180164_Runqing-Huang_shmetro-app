import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import folium
from streamlit_folium import st_folium

# ---------- 1. Load data ----------
@st.cache_data
def load_data():
    df = pd.read_csv("上海市_上海_地铁站点_wgs84.csv")
    return df

df = load_data()

# ---------- 2. Utility: Build status graph ----------
def generate_status_graph(df, scene):
    center_lat, center_lon = 31.23, 121.47
    center_radius = 0.03
    shopping_hotspots = ["上海人民广场(地铁站)", "上海南京东路(地铁站)", "上海陆家嘴(地铁站)"]
    holiday_closure = ["上海科技馆(地铁站)", "上海豫园(地铁站)"]
    line_counts = df.groupby("sname")["rname"].nunique().to_dict()

    def classify(row):
        sname = row["sname"]
        lat, lon = map(float, row["geometry"].split(","))
        dist = np.sqrt((lat - center_lat)**2 + (lon - center_lon)**2)
        lines = line_counts.get(sname, 1)
        if scene in ["weekday_morning", "weekday_evening"]:
            if lines >= 3 and dist < center_radius:
                return "red"
            elif lines >= 2 or dist < center_radius:
                return "yellow"
            else:
                return "green"
        elif scene == "weekday_noon":
            return "yellow" if dist < center_radius else "green"
        elif scene == "weekend":
            if sname in shopping_hotspots:
                return "red"
            elif dist < center_radius:
                return "yellow"
            else:
                return "green"
        elif scene == "holiday":
            if sname in holiday_closure:
                return "black"
            elif sname in shopping_hotspots:
                return "red"
            else:
                return "yellow" if dist < center_radius else "green"
        elif scene == "night":
            return "green"
        else:
            return "green"

    df_scene = df.copy()
    df_scene["status_color"] = df_scene.apply(classify, axis=1)

    G = nx.Graph()
    for _, row in df_scene.iterrows():
        name = row["sname"]
        lat, lon = map(float, row["geometry"].split(","))
        G.add_node(name, pos=(lat, lon))

    state_cost = {"green": 1, "yellow": 5, "red": 10, "black": float("inf")}
    grouped = df_scene.groupby("rname")
    for _, group in grouped:
        group_sorted = group.sort_values("order")
        prev = None
        for _, row in group_sorted.iterrows():
            curr = row["sname"]
            if prev is not None:
                c1 = df_scene[df_scene["sname"] == prev]["status_color"].values[0]
                c2 = df_scene[df_scene["sname"] == curr]["status_color"].values[0]
                if c1 == "black" or c2 == "black":
                    continue
                weight = (state_cost[c1] + state_cost[c2]) / 2
                G.add_edge(prev, curr, weight=weight)
            prev = curr

    return G, df_scene

# ---------- 3. Streamlit UI ----------
st.title("🚇 上海地铁路径推荐系统 Metro Route Recommender")

with st.sidebar:
    st.header("选择路径参数 Select Options")
    all_stations = sorted(df["sname"].unique())
    start_station = st.selectbox("起点站 Start Station", all_stations, index=0)
    end_station = st.selectbox("终点站 End Station", all_stations, index=10)
    scene = st.radio("时间场景 Time Scenario", [
        "weekday_morning", "weekday_evening", "weekday_noon",
        "weekend", "holiday", "night"])

# 使用 session_state 来保持提交状态
if "run" not in st.session_state:
    st.session_state.run = False

if st.button("🚀 推荐路径 Recommend Path"):
    st.session_state.run = True

# ---------- 4. Generate path + map ----------
if st.session_state.run:
    G, df_scene = generate_status_graph(df, scene)
    try:
        path = nx.dijkstra_path(G, start_station, end_station, weight="weight")
        total_cost = nx.dijkstra_path_length(G, start_station, end_station, weight="weight")
        st.success(f"共找到 {len(path)} 个站点，总成本 {total_cost:.2f} ✅")
        st.markdown("**推荐路径如下：**")
        for station in path:
            st.write("➡️", station)
    except nx.NetworkXNoPath:
        st.error("未找到可通路径（部分站点可能关闭）")
        path = []

    # 绘制地图
    m = folium.Map(location=[31.23, 121.47], zoom_start=12)
    for _, row in df_scene.iterrows():
        lat, lon = map(float, row["geometry"].split(","))
        color = row["status_color"]
        folium.CircleMarker(
            location=(lat, lon),
            radius=3,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.5
        ).add_to(m)

    if path:
        coords = [G.nodes[n]["pos"] for n in path]
        folium.PolyLine(coords, color="orange", weight=5).add_to(m)
        folium.Marker(coords[0], popup="Start", icon=folium.Icon(color="green")).add_to(m)
        folium.Marker(coords[-1], popup="End", icon=folium.Icon(color="red")).add_to(m)

    st.markdown("---")
    st.subheader("🗺️ 路径地图 Map")
    st_folium(m, width=750, height=500)
