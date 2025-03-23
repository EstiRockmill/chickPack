import { CommonModule } from '@angular/common';
import { Component, OnInit } from '@angular/core';
import { Router, RouterLink } from '@angular/router';
import { ShowRoomsService } from '../../../Service/show-rooms.service';
@Component({
  selector: 'app-show-rooms',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './show-rooms.component.html',
  styleUrl: './show-rooms.component.css'
})
export class ShowRoomsComponent implements OnInit{
  loading = false;
  error = '';
  rooms:any[] = []
  


  public constructor(private router: Router,private showRooms :ShowRoomsService  ){}


  ngOnInit(): void {
    this.rooms=this.showRooms.getShowRooms()
  }

    selectRoom(room:any): void {
    this.router.navigate(['/image-upload', room.id]);
  }
}
