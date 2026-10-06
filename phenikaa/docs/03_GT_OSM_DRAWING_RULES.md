# Quy Tac Ve GT OSM

Ban day du nam tai:

```text
docs/03_GT_OSM_DRAWING_GUIDE_FULL.md
```

Muc tieu GT hien tai la hoc cac vector nhin/duoc suy ra hop ly tu camera:

- `lane_divider`: vach phan lan that, net lien hoac dut.
- `road_edge_marking`: vach mep trang that, khong gop voi via he.
- `stop_line`: vach dung.
- `ped_crossing`: vung/vach nguoi di bo qua duong.
- `boundary`: bo via, mep ket thuc mat duong, ranh gioi vat ly.
- `speed_bump`: go giam toc, polygon la chinh.
- `centerline`: line ao ho tro topology/phan tich, khong nhat thiet dua vao train MapTR neu model dang train 6 class noi tren.

Rule quan trong: nhung line ao phuc vu topology nhu connector trong giao lo nen tach semantic ro, khong tron voi vach that neu muc tieu la hoc tu camera.
