using AutoMapper;
using EasyPack.API.Models.Box;
using EasyPack.API.Models.Category;
using EasyPack.API.Models.Item;
using EasyPack.API.Models.Room;
using EasyPack.API.Models.User;
using EasyPack.Core.Models;

namespace EasyPack.API.Mapping
{
    public class PostModelsMappingProfile : Profile
    {
        public PostModelsMappingProfile()
        {
            CreateMap<CategoryPostModel, Category>();
            CreateMap<BoxPostModel, Box>();
            CreateMap<RoomPostModel, Room>();
            CreateMap<RoomPutModel, Room>();
            CreateMap<UserPostModel, User>();
            CreateMap<ItemPostModel, Item>();
            CreateMap<BoxPutModel, Box>();
            CreateMap<ItemPutModel, Item>();
        }
    }
}
