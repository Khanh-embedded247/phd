# 06. Thesis QA

## Dong gop cua project

- Dua du lieu Phenikaa 12 camera vao pipeline MapTR.
- Chuan hoa GT OSM theo semantic class HD map.
- Sua dataset/loader de hoc 6 class perception.
- Sua frame ego/base_link thay vi mac dinh lidar_top.
- Export prediction sang OSM graph va postprocess spatial graph thanh HD map mong.
- Tao visualization professional de danh gia dinh tinh.

## Vi sao khong train `virtual_*`?

`virtual_*` la logic/topology do nguoi ve, nhieu khi camera/LiDAR khong quan sat truc tiep. Giai doan perception dau tien chi train doi tuong vat ly/sensor nhin duoc.

## Vi sao can postprocess?

Model predict theo patch/frame, nen cung mot vach xuat hien nhieu lan va lech nho. Postprocess gom prediction cung class thanh graph mong, giam roi va tranh noi cheo giao lo.

## Vi sao van tao LiDAR synced du camera-only?

De giu format dataset, QA/visualize, va mo rong fusion sau nay. Voi camera-only, day la cache co the tao lai.
