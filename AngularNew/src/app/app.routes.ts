import { Routes } from '@angular/router';
import { SignInComponent } from './Components/User/sign-in/sign-in.component';
import { SignUpComponent } from './Components/User/sign-up/sign-up.component';
import { ShowRoomsComponent } from './Components/Room/show-rooms/show-rooms.component';
import { HomePageComponent } from './Components/Home/home-page/home-page.component';
import { ImageUploadComponent } from './Components/Room/image-upload/image-upload.component';
import { RoomPackComponent } from './Components/Room/room-pack/room-pack.component';
// import { ImageUploadComponent } from './Components/Room/image-upload/image-upload.component';


export const routes: Routes = [
{path:'',component:HomePageComponent},//default ,
{path:'login',component:SignInComponent},
{path:'signUp',component:SignUpComponent},
{ 
    path: 'room-selection', 
    component: ShowRoomsComponent, 
    // canActivate: [AuthGuard] 
  },
  { 
    path: 'image-upload/:roomId', 
    component: RoomPackComponent, 
    // canActivate: [AuthGuard] 
  }

];
