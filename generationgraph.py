import matplotlib.pyplot as plt

load_gt = [
    29250.5, 28493.5, 27424.6, 26714.8, 26497.9, 26923, 26978.6, 27849.2,
    31444.1, 33192.1, 34296.9, 34592.4, 34414.3, 34234.9, 34993.3, 35083.3,
    35087.2, 34897.2, 33210.1, 32211.5, 31232, 30342.2, 29534.1, 29202.9
]

load_predict = [
    30434.8548395157, 29790.0706389562, 28629.3315154321, 28424.3572367165,
    28404.7587171188, 28063.8854357105, 27995.8977688628, 31899.2684247614,
    37029.3065761327, 39547.3272348668, 39699.4570850201, 39495.1699836861,
    39625.1301308063, 39736.322085703, 39778.7206273936, 39433.4296729278,
    38570.350197691, 36772.9676782485, 34172.167459725, 32144.9146519607,
    31287.1834496692, 30643.5830753472, 30029.1212996712, 28995.8150130434
]

pv_gt = [0, 0, 0, 0, 0, 0, 0, 5.39999996, 92.2999988, 70.3000003, 245.500003, 297.199998, 271.1, 271.699995, 298.0999984, 540.699998, 333.9999954, 77.8999973, 15.19999994, 0.9, 0, 0, 0, 0]
pv_predict = [0, 0, 0, 0, 0, 0, 0, 0, 173.9727808, 180.0563939, 185.1389643, 193.0415162, 191.8326627, 201.498405, 196.8958704, 630.76776172, 198.9560808, 173.0473803, 0, 0, 0, 0, 0, 0]


# X 축 설정 (1시간 단위의 하루)
hours = list(range(24))

# 그래프 그리기
plt.figure(figsize=(10, 6))
plt.plot(hours, load_gt, marker='o', label='Ground Truth', color='blue')
plt.plot(hours, load_predict, marker='x', label='Prediction', color='red')
plt.title('Load Ground Truth vs. Prediction')
plt.xlabel('Hour')
plt.ylabel('Value')
plt.xticks(hours)  # x축 레이블 설정
plt.ylim(0, 70000)
plt.legend()  # 범례 표시
plt.grid(True)
plt.tight_layout()
plt.savefig('images/load_comparison_plot.png')
# plt.show()


# 그래프 그리기
plt.figure(figsize=(10, 6))
plt.plot(hours, pv_gt, marker='o', label='Ground Truth', color='blue')
plt.plot(hours, pv_predict, marker='x', label='Prediction', color='red')
plt.title('PV Ground Truth vs. Prediction')
plt.xlabel('Hour')
plt.ylabel('Value')
plt.xticks(hours)  # x축 레이블 설정
plt.ylim(0, 2000)
plt.legend()  # 범례 표시
plt.grid(True)
plt.tight_layout()
plt.savefig('images/pv_comparison_plot.png')
plt.show()