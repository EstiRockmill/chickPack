import { Injectable } from '@angular/core';
@Injectable({
  providedIn: 'root'
})
export class ShowRoomsService {

  constructor() { }
  // rooms = [

  //   {
  //     id: '1',
  //     name: 'Living Room',
  //     description: 'Pack your living room items efficiently',
  //     imageUrl: '../../assests/pictures/livingRoom.avif'
  //   },
  //   {
  //     id: '2',
  //     name: 'Bedroom',
  //     description: 'Optimize your bedroom packing',
  //     imageUrl: '../../../../assests/pictures/BedRoom.png'
  //   },
  //   {
  //     id: '3',
  //     name: 'Kitchen',
  //     description: 'Pack your kitchen items safely',
  //     imageUrl: 'src/pictures/kitchen.avif'
  //   },
  //   {
  //     id: '4',
  //     name: 'Office',
  //     description: 'Organize your office equipment',
  //     imageUrl: 'src/pictures/office.avif'
  //   },
  //   {
  //     id: '5',
  //     name: 'Garage',
  //     description: 'Pack your garage items efficiently',
  //     imageUrl: 'src/pictures/garage.avif'
  //   },
  //   {
  //     id: '6',
  //     name: 'Custom Room',
  //     description: 'Define your own room type',
  //     imageUrl: 'src/pictures/customRoom.avif'
  //   }
  // ];
  rooms = [
    {
      id: '1',
      name: 'Living Room',
      description: 'Pack your living room items efficiently',
      imageUrl: 'https://images.unsplash.com/photo-1513694203232-719a280e022f?ixlib=rb-1.2.1&auto=format&fit=crop&w=500&q=60'
    },
    {
      id: '2',
      name: 'Bedroom',
      description: 'Optimize your bedroom packing',
      imageUrl: 'https://images.unsplash.com/photo-1522771739844-6a9f6d5f14af?ixlib=rb-1.2.1&auto=format&fit=crop&w=500&q=60'
    },
    {
      id: '3',
      name: 'Kitchen',
      description: 'Pack your kitchen items safely',
      imageUrl: 'https://images.unsplash.com/photo-1556911220-bff31c812dba?ixlib=rb-1.2.1&auto=format&fit=crop&w=500&q=60'
    },
    {
      id: '4',
      name: 'Office',
      description: 'Organize your office equipment',
      imageUrl: 'https://images.unsplash.com/photo-1524758631624-e2822e304c36?ixlib=rb-1.2.1&auto=format&fit=crop&w=500&q=60'
    },
    {
      id: '5',
      name: 'Garage',
      description: 'Pack your garage items efficiently',
      imageUrl: 'https://images.unsplash.com/photo-1558402529-d2638a7023e9?ixlib=rb-1.2.1&auto=format&fit=crop&w=500&q=60'
    },
    {
      id: '6',
      name: 'Custom Room',
      description: 'Define your own room type',
      imageUrl: 'https://images.unsplash.com/photo-1513519245088-0e12902e5a38?ixlib=rb-1.2.1&auto=format&fit=crop&w=500&q=60'
    }
  ];
  getShowRooms() {
    return this.rooms;
  }
}
